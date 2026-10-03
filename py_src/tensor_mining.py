"""
Phase 3 "node-level candidate-gene batching" backend (see
.temp/parallel_improvement_plan.md section 6, and the follow-up scope
decision to batch across candidate genes within one tree node rather than
rewrite the whole mining engine to breadth-first).

buildProbabilityTreeOnTargetGene (src/fbn_tree.cpp) recursion is kept
*identical* here - same branching, same essential-gene/correlation logic,
same string-built rule identities. The only thing replaced is the innermost
O(candidates) loop (getGeneProbabilities_measurements -> getGeneProbabilities
-> getGeneProbabilities_basic -> getBasicMeasures -> up to 12 matchCount
calls per candidate gene), which is now computed for ALL remaining candidate
genes at a node in one batched NumPy pass instead of one C++ call per gene.

Derivation summary (see getBasicMeasures/getGeneProbabilities_basic in
src/fbn_core.cpp for the ground truth being replicated):

- At a tree node, the conditional-gene set is `existing_fixed_genes` (already
  decided T/F along this tree path) plus exactly one `candidate` gene being
  tried. matchCount(m, pattern) is a conjunction (AND) over all rows, which
  is commutative - so instead of concatenating rows into one `m` matrix per
  candidate, we can split the conjunction into an "active mask" term (from
  the existing-fixed genes, identical for every candidate at this node) and
  a per-candidate term, then batch the per-candidate term as one
  matrix-vector product (candidates x columns) @ (columns,) per contingency
  cell - a GEMV instead of N separate matchCount calls.
- target_T_count/target_F_count reduce algebraically (summing the two
  candidate-states of a 2-row matchCount always marginalizes out the
  candidate's own value) to "count of columns where the target row equals
  1/0 AND the column is not a sentinel (value 9) sample-boundary column"
  over the full column range for that temporal step - independent of the
  candidate or any existing fixed genes. This holds in both the size==1 and
  size>1 branches of the original C++ (the size>1 branch's "last
  conditional gene" recount is equivalent, since the sentinel value is
  uniform across *every* gene row in a boundary column - any single row,
  including the target row itself, detects it), so one formula covers both.

The advanced-measures stage (Fisher exact test, chi-square, entropy/mutual
information, essential-gene/bestFit scoring - src/fbn_core.cpp's
getAdvancedMeasures) is cheap per-candidate (O(1), not O(columns)) and is
reused UNCHANGED via the existing fbnnet_core.getAdvancedMeasures binding, so
none of that already-validated logic is reimplemented here.
"""
from typing import Any, Dict, List, Optional

import numpy as np

import fbnnet_core
import fbnnet_tree


def _slice_step(full_matrix: np.ndarray, i: int):
    """Replicates generate_temporal_gene_states's column slicing for temporal step i (0-indexed)."""
    n_state = full_matrix.shape[1] - 1
    start_col = i + 1
    end_col = n_state + 1
    current = full_matrix[:, start_col:end_col]
    previous = full_matrix[:, 0:n_state - i]
    return current, previous


def batched_basic_measures(
    main_parameters: Dict[str, Any],
    target_gene: str,
    candidate_genes: List[str],
    fixed_state: Dict[str, int],
    temporal: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Vectorized equivalent of calling fbnnet_core.getGeneProbabilities_basic
    once per candidate gene (with `new_conditional_gene=[candidate]` and the
    same `fixedgenestate=fixed_state`), for ALL `candidate_genes` at once.

    Returns {gene: {str(timestep): basic_measures_dict, ...}, ...} - same
    shape/keys as getGeneProbabilities_basic's return value, per candidate.

    Assumes (matching how this is actually called from the tree builder)
    that none of `candidate_genes` are already keys in `fixed_state` - the
    "candidate already fixed" rebalancing branch in the original
    getGeneProbabilities_basic never triggers for genes drawn from the
    tree's unprocessed-gene list.
    """
    rownames = main_parameters["rownames"]
    row_index = {g: idx for idx, g in enumerate(rownames)}
    total_samples = main_parameters["total_samples"]
    n_timepoints = main_parameters["total_timepoints"]
    current_states = main_parameters["currentStates"]

    existing_genes = list(fixed_state.keys())
    existing_idx = np.array([row_index[g] for g in existing_genes], dtype=np.int64)
    existing_bits = np.array([fixed_state[g] for g in existing_genes], dtype=np.float64)
    target_row_idx = row_index[target_gene]
    candidate_idx = np.array([row_index[g] for g in candidate_genes], dtype=np.int64)
    num_of_conditional_genes = len(existing_genes) + 1

    per_gene_results: Dict[str, Dict[str, Dict[str, Any]]] = {g: {} for g in candidate_genes}

    for i in range(temporal):
        full_matrix = np.asarray(current_states[i])
        cur, prev = _slice_step(full_matrix, i)
        width = cur.shape[1]

        target_cur = cur[target_row_idx]
        target_prev = prev[target_row_idx]

        active_mask = np.ones(width, dtype=np.float64)
        active_mask_c = np.ones(width, dtype=np.float64)
        for k in range(len(existing_genes)):
            gidx = existing_idx[k]
            bit = existing_bits[k]
            active_mask = active_mask * (prev[gidx] == bit)
            active_mask_c = active_mask_c * (cur[gidx] == bit)

        AT1 = active_mask * (target_cur == 1)
        AT0 = active_mask * (target_cur == 0)
        AC1 = active_mask_c * (target_prev == 1)
        AC0 = active_mask_c * (target_prev == 0)

        candidates_prev = prev[candidate_idx]
        candidates_cur = cur[candidate_idx]

        cand_prev_1 = (candidates_prev == 1).astype(np.float64)
        cand_prev_0 = (candidates_prev == 0).astype(np.float64)
        cand_cur_1 = (candidates_cur == 1).astype(np.float64)
        cand_cur_0 = (candidates_cur == 0).astype(np.float64)

        count_cond_T_target_T = cand_prev_1 @ AT1
        count_cond_F_target_T = cand_prev_0 @ AT1
        count_cond_T_target_F = cand_prev_1 @ AT0
        count_cond_F_target_F = cand_prev_0 @ AT0

        count_cond_T_target_T_c = cand_cur_1 @ AC1
        count_cond_F_target_T_c = cand_cur_1 @ AC0
        count_cond_T_target_F_c = cand_cur_0 @ AC1
        count_cond_F_target_F_c = cand_cur_0 @ AC0

        cond_T_count = count_cond_T_target_T + count_cond_T_target_F
        cond_F_count = count_cond_F_target_T + count_cond_F_target_F
        cond_T_count_c = count_cond_T_target_T_c + count_cond_T_target_F_c
        cond_F_count_c = count_cond_F_target_T_c + count_cond_F_target_F_c

        # Boundary columns between concatenated samples are sentinel-filled (value 9)
        # uniformly across *all* gene rows, so any single row can detect them; the C++
        # side's "last conditional gene" recount is equivalent to this for that reason.
        prev_valid = target_prev != 9
        target_T_count = int(np.count_nonzero((target_cur == 1) & prev_valid))
        target_F_count = int(np.count_nonzero((target_cur == 0) & prev_valid))

        time_step = i + 1
        total_calculated_timepoints = n_timepoints - (total_samples * time_step)
        time_step_key = str(time_step)

        def _ints(a):
            return np.rint(a).astype(np.int64).tolist()

        cols = {
            "cond_T_count": _ints(cond_T_count),
            "cond_F_count": _ints(cond_F_count),
            "cond_T_count_c": _ints(cond_T_count_c),
            "cond_F_count_c": _ints(cond_F_count_c),
            "count_cond_T_target_T": _ints(count_cond_T_target_T),
            "count_cond_F_target_T": _ints(count_cond_F_target_T),
            "count_cond_T_target_F": _ints(count_cond_T_target_F),
            "count_cond_F_target_F": _ints(count_cond_F_target_F),
            "count_cond_T_target_T_c": _ints(count_cond_T_target_T_c),
            "count_cond_F_target_T_c": _ints(count_cond_F_target_T_c),
            "count_cond_T_target_F_c": _ints(count_cond_T_target_F_c),
            "count_cond_F_target_F_c": _ints(count_cond_F_target_F_c),
        }

        for j, gene in enumerate(candidate_genes):
            record = {"target_T_count": target_T_count, "target_F_count": target_F_count}
            for name, values in cols.items():
                record[name] = values[j]
            record["total_calculated_timepoints"] = total_calculated_timepoints
            record["num_of_conditional_genes"] = num_of_conditional_genes
            record["timestep"] = time_step
            per_gene_results[gene][time_step_key] = record

    return per_gene_results


def gene_probabilities_for_candidates(
    main_parameters: Dict[str, Any],
    target_gene: str,
    candidate_genes: List[str],
    fixed_state: Dict[str, int],
    temporal: int,
    show_basic_measures: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Vectorized equivalent of calling fbnnet_core.getGeneProbabilities once
    per candidate gene. Reuses the existing (unchanged) C++
    getAdvancedMeasures for the per-candidate scalar statistics stage (cheap,
    O(1) per candidate) - only the O(columns) counting stage is batched.

    Returns {gene: {"BestFitP": ..., "BestFitN": ...}, ...}.
    """
    basic = batched_basic_measures(main_parameters, target_gene, candidate_genes, fixed_state, temporal)

    result: Dict[str, Dict[str, Any]] = {}
    for gene in candidate_genes:
        result_group = basic[gene]
        best_fit_p = None
        best_fit_n = None
        for time_step_key, basic_measures in result_group.items():
            advanced = fbnnet_core.getAdvancedMeasures(basic_measures, show_basic_measures)
            timestep = advanced["timestep"]
            if best_fit_p is None:
                best_fit_p = advanced
                best_fit_n = advanced
                continue
            if best_fit_p["bestFitP"] > advanced["bestFitP"]:
                best_fit_p = advanced
            if best_fit_n["bestFitN"] > advanced["bestFitN"]:
                best_fit_n = advanced
            if best_fit_p["bestFitP"] == advanced["bestFitP"] and best_fit_p["timestep"] > timestep:
                best_fit_p = advanced
            if best_fit_n["bestFitN"] == advanced["bestFitN"] and best_fit_n["timestep"] > timestep:
                best_fit_n = advanced
        result[gene] = {"BestFitP": best_fit_p, "BestFitN": best_fit_n}
    return result


def gene_probabilities_measurements(
    main_parameters: Dict[str, Any],
    target_gene: str,
    genes: List[str],
    fixed_state: Dict[str, int],
    temporal: int,
    show_basic_measures: bool = False,
) -> "dict":
    """
    Vectorized equivalent of fbnnet_tree.getGeneProbabilities_measurements:
    tries every gene in `genes` as the next conditional gene (batched), then
    applies the same essential-gene/correlation filtering, preserving the
    input order of `genes` for the result's key order (matches the C++
    dict's insertion order, which recursion relies on).
    """
    if not genes:
        return {}

    probabilities = gene_probabilities_for_candidates(
        main_parameters, target_gene, genes, fixed_state, temporal, show_basic_measures
    )

    result: Dict[str, Any] = {}
    for gene in genes:
        combo = probabilities[gene]
        p = combo["BestFitP"]
        n = combo["BestFitN"]

        if not p["is_essential_gene"] and not n["is_essential_gene"]:
            continue

        if not p["is_essential_gene"]:
            p = n
        if not n["is_essential_gene"]:
            n = p

        if not p["isPositiveCorrelated"] and not p["isNegativeCorrelated"]:
            p = n
        if not n["isPositiveCorrelated"] and not n["isNegativeCorrelated"]:
            if p == n:
                continue
            n = p

        result[gene] = {
            "probabilityOfFourCombines_P": p,
            "probabilityOfFourCombines_N": n,
        }

    return result


def _cpp_double_str(x: float) -> str:
    """Matches C++ std::to_string(double): fixed notation, 6 fractional digits."""
    return f"{float(x):.6f}"


def _cpp_bool_str(b: bool) -> str:
    """Matches C++ std::to_string(bool): bool implicitly converts to int (0/1)."""
    return "1" if b else "0"


def build_probability_tree_on_target_gene(
    target_gene: str,
    main_parameters: Dict[str, Any],
    genes: List[str],
    matched_genes: Optional[Dict[str, int]],
    matched_expression: Optional[List[str]],
    max_k: int,
    temporal: int,
    show_basic_measures: bool,
    find_positive_regulate: bool,
    find_negative_regulate: bool,
) -> Dict[str, Any]:
    """
    Python port of buildProbabilityTreeOnTargetGene (src/fbn_tree.cpp), with
    the innermost per-node "try every candidate gene" loop replaced by the
    batched `gene_probabilities_measurements` above. Recursion/branching
    control flow, string-built rule identities, and Activator_of/Inhibitor_of
    record construction are kept IDENTICAL to the C++ original.
    """
    fixed_state = matched_genes if matched_genes is not None else {}
    measurements = gene_probabilities_measurements(
        main_parameters, target_gene, genes, fixed_state, temporal, show_basic_measures
    )

    new_genes = list(measurements.keys())
    unprocessed_genes = list(new_genes)

    if max_k > len(unprocessed_genes):
        max_k = len(unprocessed_genes)

    result: Dict[str, Any] = {}

    for gene in new_genes:
        pmax_k = max_k

        unprocessed_genes = [g for g in unprocessed_genes if g != gene]

        new_matched_genes_t: Dict[str, int] = {}
        new_matched_genes_f: Dict[str, int] = {}
        preprocessed: List[str] = []

        if matched_genes is not None:
            for k, v in matched_genes.items():
                new_matched_genes_t[k] = v
                new_matched_genes_f[k] = v
                preprocessed.append(k)

        if matched_expression is not None:
            expression_t = list(matched_expression) + ["&", gene]
            expression_f = list(matched_expression) + ["&", "!", gene]
            new_matched_genes_t.setdefault(gene, 1)
            new_matched_genes_f.setdefault(gene, 0)
        else:
            new_matched_genes_t[gene] = 1
            new_matched_genes_f[gene] = 0
            expression_t = [gene]
            expression_f = ["!", gene]

        exp_t = "".join(expression_t)
        exp_f = "".join(expression_f)

        input_genes = sorted(new_matched_genes_t.keys(), reverse=True)

        if preprocessed and gene in preprocessed:
            result[gene] = {}
            continue

        measurement = measurements[gene]
        p = measurement["probabilityOfFourCombines_P"]
        n = measurement["probabilityOfFourCombines_N"]

        best_fit_p = p["bestFitP"]
        is_essential_gene_p = p["is_essential_gene"]
        best_fit_n = n["bestFitN"]
        is_essential_gene_n = n["is_essential_gene"]
        sign_p = p["signal_sign_T"]
        sign_n = n["signal_sign_F"]

        subresult_t: Dict[str, Any] = {}
        subresult_f: Dict[str, Any] = {}

        if pmax_k > 1 and not (find_positive_regulate and find_negative_regulate):
            find_positive_regulate = find_positive_regulate or best_fit_p == 0
            find_negative_regulate = find_negative_regulate or best_fit_n == 0
            pmax_k -= 1

            excluded_subgenes = set(input_genes)

            next_genes_t = [g for g in unprocessed_genes if g not in excluded_subgenes]
            if next_genes_t and is_essential_gene_p and (best_fit_p > 0 or best_fit_n > 0):
                subresult_t = build_probability_tree_on_target_gene(
                    target_gene, main_parameters, next_genes_t, new_matched_genes_t, [exp_t],
                    pmax_k, temporal, show_basic_measures, find_positive_regulate, find_negative_regulate,
                )

            next_genes_f = [g for g in unprocessed_genes if g not in excluded_subgenes]
            if next_genes_f and is_essential_gene_n and (best_fit_p > 0 or best_fit_n > 0):
                subresult_f = build_probability_tree_on_target_gene(
                    target_gene, main_parameters, next_genes_f, new_matched_genes_f, [exp_f],
                    pmax_k, temporal, show_basic_measures, find_positive_regulate, find_negative_regulate,
                )

        pick_t_support = p["pickT_support"]
        pick_t_causality_test = p["pickT_causality_test"]
        pick_t_value = p["Signal_P"]
        pick_t_noise = p["Noise_P"]
        pick_t_confidence_counter = p["pickT_confidenceCounter"]
        pick_t_all_confidence = p["pickT_all_confidence"]
        pick_t_max_confidence = p["pickT_max_confidence"]
        is_negative_correlated_t = p["isNegativeCorrelated"]
        is_positive_correlated_t = p["isPositiveCorrelated"]
        timestep_t = p["timestep"]
        best_fit_p_val = p["bestFitP"]
        p_value_p = p["p_value"]
        pick_t_mutual_info = p["pickT_mutualInfo"]

        pick_f_support = n["pickF_support"]
        pick_f_causality_test = n["pickF_causality_test"]
        pick_f_value = n["Signal_N"]
        pick_f_noise = n["Noise_N"]
        pick_f_confidence_counter = n["pickF_confidenceCounter"]
        pick_f_all_confidence = n["pickF_all_confidence"]
        pick_f_max_confidence = n["pickF_max_confidence"]
        is_negative_correlated_f = n["isNegativeCorrelated"]
        is_positive_correlated_f = n["isPositiveCorrelated"]
        timestep_f = n["timestep"]
        best_fit_n_val = n["bestFitN"]
        p_value_n = n["p_value"]
        pick_f_mutual_info = n["pickF_mutualInfo"]

        pick_exp_t = ""
        pick_exp_f = ""
        identity_t = ""
        identity_f = ""

        if sign_p == "TT":
            pattern = [f"{ig}${new_matched_genes_t[ig]}" for ig in input_genes]
            identity_t = "_".join(pattern)
            pick_exp_t = exp_t
        elif sign_p == "TF":
            pattern = [f"{ig}${new_matched_genes_f[ig]}" for ig in input_genes]
            identity_t = "_".join(pattern)
            pick_exp_t = exp_f

        identity_t = "_".join([identity_t, "Activator_of", target_gene])

        activator = {
            "factor": pick_exp_t,
            "Confidence": _cpp_double_str(pick_t_value),
            "ConfidenceCounter": _cpp_double_str(pick_t_confidence_counter),
            "all_confidence": _cpp_double_str(pick_t_all_confidence),
            "max_confidence": _cpp_double_str(pick_t_max_confidence),
            "support": _cpp_double_str(pick_t_support),
            "causality_test": _cpp_double_str(pick_t_causality_test),
            "Noise": _cpp_double_str(pick_t_noise),
            "Identity": identity_t,
            "type": sign_p,
            "timestep": str(int(timestep_t)),
            "isNegativeCorrelated": _cpp_bool_str(is_negative_correlated_t),
            "isPositiveCorrelated": _cpp_bool_str(is_positive_correlated_t),
            "bestFitP": _cpp_double_str(best_fit_p_val),
            "p_value": _cpp_double_str(p_value_p),
            "mutualInfo": _cpp_double_str(pick_t_mutual_info),
        }

        if sign_n == "FT":
            pattern = [f"{ig}${new_matched_genes_t[ig]}" for ig in input_genes]
            identity_f = "_".join(pattern)
            pick_exp_f = exp_t
        elif sign_n == "FF":
            pattern = [f"{ig}${new_matched_genes_f[ig]}" for ig in input_genes]
            identity_f = "_".join(pattern)
            pick_exp_f = exp_f

        identity_f = "_".join([identity_f, "Inhibitor_of", target_gene])

        inhibitor = {
            "factor": pick_exp_f,
            "Confidence": _cpp_double_str(pick_f_value),
            "ConfidenceCounter": _cpp_double_str(pick_f_confidence_counter),
            "all_confidence": _cpp_double_str(pick_f_all_confidence),
            "max_confidence": _cpp_double_str(pick_f_max_confidence),
            "support": _cpp_double_str(pick_f_support),
            "causality_test": _cpp_double_str(pick_f_causality_test),
            "Noise": _cpp_double_str(pick_f_noise),
            "Identity": identity_f,
            "type": sign_n,
            "timestep": str(int(timestep_f)),
            "isNegativeCorrelated": _cpp_bool_str(is_negative_correlated_f),
            "isPositiveCorrelated": _cpp_bool_str(is_positive_correlated_f),
            "bestFitN": _cpp_double_str(best_fit_n_val),
            "p_value": _cpp_double_str(p_value_n),
            "mutualInfo": _cpp_double_str(pick_f_mutual_info),
        }

        in_res: Dict[str, Any] = {
            "ActivatorAndInhibitor": {"Activator": activator, "Inhibitor": inhibitor},
            "Input": input_genes,
        }

        if subresult_t:
            in_res["SubGenesT"] = subresult_t
        if subresult_f:
            in_res["SubGenesF"] = subresult_f

        result[gene] = in_res

    return result


def process_cube_algorithm(
    target_gene: str,
    conditional_genes: List[str],
    max_k: int,
    temporal: int,
    main_parameters: Dict[str, Any],
    matched_genes: Optional[Dict[str, int]] = None,
    matched_expression: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Python port of process_cube_algorithm (src/fbn_tree.cpp) using the
    node-level batched tensor-mining backend."""
    sub_genes = build_probability_tree_on_target_gene(
        target_gene, main_parameters, conditional_genes, matched_genes, matched_expression,
        max_k, temporal, False, False, False,
    )
    return {target_gene: {"SubGenes": sub_genes}}
