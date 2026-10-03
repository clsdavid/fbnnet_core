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
  candidate's own value) to simply "count of columns where the target row
  equals 1/0" over the full unmasked column range for that temporal step -
  independent of the candidate or any existing fixed genes. This holds in
  both the size==1 and size>1 branches of the original C++, so one formula
  covers both.

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

        target_T_count = int(np.count_nonzero(target_cur == 1))
        target_F_count = int(np.count_nonzero(target_cur == 0))

        time_step = i + 1
        total_calculated_timepoints = n_timepoints - (total_samples * time_step)
        time_step_key = str(time_step)

        for j, gene in enumerate(candidate_genes):
            per_gene_results[gene][time_step_key] = {
                "target_T_count": target_T_count,
                "target_F_count": target_F_count,
                "cond_T_count": int(round(cond_T_count[j])),
                "cond_F_count": int(round(cond_F_count[j])),
                "cond_T_count_c": int(round(cond_T_count_c[j])),
                "cond_F_count_c": int(round(cond_F_count_c[j])),
                "count_cond_T_target_T": int(round(count_cond_T_target_T[j])),
                "count_cond_F_target_T": int(round(count_cond_F_target_T[j])),
                "count_cond_T_target_F": int(round(count_cond_T_target_F[j])),
                "count_cond_F_target_F": int(round(count_cond_F_target_F[j])),
                "count_cond_T_target_T_c": int(round(count_cond_T_target_T_c[j])),
                "count_cond_F_target_T_c": int(round(count_cond_F_target_T_c[j])),
                "count_cond_T_target_F_c": int(round(count_cond_T_target_F_c[j])),
                "count_cond_F_target_F_c": int(round(count_cond_F_target_F_c[j])),
                "total_calculated_timepoints": total_calculated_timepoints,
                "num_of_conditional_genes": num_of_conditional_genes,
                "timestep": time_step,
            }

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
