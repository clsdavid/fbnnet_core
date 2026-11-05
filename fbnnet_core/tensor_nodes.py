# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Leshi Chen
"""Columnar evaluation of one tree node for the ``tensor`` mining backend.

`tensor_mining.gene_probabilities_measurements` used to build one Python dict per candidate gene and
call the C++ `getAdvancedMeasures` once per candidate. Profiling showed that this per-candidate
work (not the counting) dominated the run time, and that the memory traffic of the temporary
float64 matrices made parallel workers run about twice as slowly as a single one. This module keeps
the results identical but changes how they are computed:

* The 0/1 indicator matrices of every temporal step are built once (`prepare`) as ``float32`` and
  shared by forked workers copy-on-write, instead of being rebuilt as ``float64`` at every node.
* The eight contingency counts of all candidates come from one small matrix product per step.
* Candidates with identical count tuples give identical measures, so the measures are computed once
  per distinct tuple (counts are bounded by the number of columns, so tuples repeat a lot), in one
  call of the C++ `getAdvancedMeasuresBatch`, which returns arrays instead of one dict per tuple.
* The essential-gene / correlation filters and the best-fit choice across temporal steps are applied
  to arrays, and result dicts are only assembled for the genes that pass.

The measures come from the same C++ arithmetic as `getAdvancedMeasures` (the dict function and the
batch function share one implementation), so the output is bit-identical to the recursive engine. The counting is written against a small array interface (`xp`), so the same code
can run on another array library later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _core

CACHE_KEY = "_tensor_cache"


def _slice_step(full_matrix: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
    n_state = full_matrix.shape[1] - 1
    return full_matrix[:, i + 1:n_state + 1], full_matrix[:, 0:n_state - i]


def prepare(main_parameters: Dict[str, Any], temporal: int) -> Dict[str, Any]:
    """Builds (once) the float32 indicator matrices of every temporal step.

    Call this in the parent before forking workers so that they share the arrays. The result is
    stored in ``main_parameters`` under a private key and returned.
    """
    cache = main_parameters.get(CACHE_KEY)
    if cache is not None and cache["temporal"] >= temporal:
        return cache

    rownames = list(main_parameters["rownames"])
    steps = []
    for i in range(temporal):
        cur, prev = _slice_step(np.asarray(main_parameters["currentStates"][i]), i)
        n_genes, width = cur.shape
        # (gene, [state==1, state==0], column): one row pair per gene, so a gather is a single copy
        prev_pair = np.empty((n_genes, 2, width), dtype=np.float32)
        prev_pair[:, 0, :] = prev == 1
        prev_pair[:, 1, :] = prev == 0
        cur_pair = np.empty((n_genes, 2, width), dtype=np.float32)
        cur_pair[:, 0, :] = cur == 1
        cur_pair[:, 1, :] = cur == 0
        steps.append({
            "prev_pair": prev_pair,
            "cur_pair": cur_pair,
            "target_cur_1": (cur == 1),
            "target_cur_0": (cur == 0),
            "target_prev_1": (prev == 1),
            "target_prev_0": (prev == 0),
            "prev_valid": (prev != 9),
            "width": width,
        })
    cache = {
        "temporal": temporal,
        "row_index": {g: idx for idx, g in enumerate(rownames)},
        "steps": steps,
        "total_samples": main_parameters["total_samples"],
        "n_timepoints": main_parameters["total_timepoints"],
    }
    main_parameters[CACHE_KEY] = cache
    return cache


def count_arrays(
    cache: Dict[str, Any],
    target_gene: str,
    candidate_genes: Sequence[str],
    fixed_state: Dict[str, int],
    step: int,
) -> Dict[str, Any]:
    """The eight contingency counts of every candidate for one temporal step, as int arrays.

    Returns ``counts`` (n x 8, column order given by COUNT_COLUMNS) and the per-node constants
    ``target_T_count`` and ``target_F_count``.
    """
    row_index = cache["row_index"]
    data = cache["steps"][step]
    prev_pair, cur_pair = data["prev_pair"], data["cur_pair"]
    target = row_index[target_gene]

    active_prev = np.ones(data["width"], dtype=np.float32)
    active_cur = np.ones(data["width"], dtype=np.float32)
    for gene, bit in fixed_state.items():
        g = row_index[gene]
        which = 0 if bit == 1 else 1
        active_prev *= prev_pair[g, which]
        active_cur *= cur_pair[g, which]

    t_cur_1 = data["target_cur_1"][target].astype(np.float32)
    t_cur_0 = data["target_cur_0"][target].astype(np.float32)
    t_prev_1 = data["target_prev_1"][target].astype(np.float32)
    t_prev_0 = data["target_prev_0"][target].astype(np.float32)
    at = np.stack([active_prev * t_cur_1, active_prev * t_cur_0], axis=1)  # AT1, AT0
    ac = np.stack([active_cur * t_prev_1, active_cur * t_prev_0], axis=1)  # AC1, AC0

    index = np.fromiter((row_index[g] for g in candidate_genes), dtype=np.int64, count=len(candidate_genes))
    n = len(index)
    if n == prev_pair.shape[0] and np.array_equal(index, np.arange(n)):
        prev_sel, cur_sel = prev_pair, cur_pair  # whole matrix in order: no gather copy
    else:
        prev_sel, cur_sel = prev_pair[index], cur_pair[index]
    prev_counts = (prev_sel.reshape(2 * n, -1) @ at).reshape(n, 2, 2)  # [cand, cond T/F, target T/F]
    cur_counts = (cur_sel.reshape(2 * n, -1) @ ac).reshape(n, 2, 2)    # [cand, cond-c T/F, target-c T/F]

    counts = np.empty((n, 8), dtype=np.int64)
    counts[:, 0] = np.rint(prev_counts[:, 0, 0])  # count_cond_T_target_T
    counts[:, 1] = np.rint(prev_counts[:, 1, 0])  # count_cond_F_target_T
    counts[:, 2] = np.rint(prev_counts[:, 0, 1])  # count_cond_T_target_F
    counts[:, 3] = np.rint(prev_counts[:, 1, 1])  # count_cond_F_target_F
    counts[:, 4] = np.rint(cur_counts[:, 0, 0])   # count_cond_T_target_T_c
    counts[:, 5] = np.rint(cur_counts[:, 0, 1])   # count_cond_F_target_T_c
    counts[:, 6] = np.rint(cur_counts[:, 1, 0])   # count_cond_T_target_F_c
    counts[:, 7] = np.rint(cur_counts[:, 1, 1])   # count_cond_F_target_F_c

    prev_valid = data["prev_valid"][target]
    return {
        "counts": counts,
        "target_T_count": int(np.count_nonzero(data["target_cur_1"][target] & prev_valid)),
        "target_F_count": int(np.count_nonzero(data["target_cur_0"][target] & prev_valid)),
    }


def _basic_measures(row: Sequence[int], constants: Dict[str, int]) -> Dict[str, Any]:
    """The dict `getAdvancedMeasures` expects (same keys as the recursive engine's basic measures)."""
    tt, ft_, tf, ff, tt_c, ft_c, tf_c, ff_c = (int(v) for v in row)
    return {
        "target_T_count": constants["target_T_count"],
        "target_F_count": constants["target_F_count"],
        "cond_T_count": tt + tf,
        "cond_F_count": ft_ + ff,
        "cond_T_count_c": tt_c + tf_c,
        "cond_F_count_c": ft_c + ff_c,
        "count_cond_T_target_T": tt,
        "count_cond_F_target_T": ft_,
        "count_cond_T_target_F": tf,
        "count_cond_F_target_F": ff,
        "count_cond_T_target_T_c": tt_c,
        "count_cond_F_target_T_c": ft_c,
        "count_cond_T_target_F_c": tf_c,
        "count_cond_F_target_F_c": ff_c,
        "total_calculated_timepoints": constants["total_calculated_timepoints"],
        "num_of_conditional_genes": constants["num_of_conditional_genes"],
        "timestep": constants["timestep"],
    }


def _unique_rows(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(distinct rows, inverse index) of an int matrix; rows are compared as raw bytes."""
    if counts.shape[0] == 0:
        return counts, np.zeros(0, dtype=np.int64)
    if counts.max(initial=0) < (1 << 15):
        packed = np.empty((counts.shape[0], 2), dtype=np.uint64)
        shifts = (np.arange(4, dtype=np.uint64) * np.uint64(16))[None, :]
        low = counts[:, :4].astype(np.uint64) << shifts
        high = counts[:, 4:].astype(np.uint64) << shifts
        packed[:, 0] = low.sum(axis=1, dtype=np.uint64)
        packed[:, 1] = high.sum(axis=1, dtype=np.uint64)
        view = np.ascontiguousarray(packed).view(np.dtype((np.void, 16))).ravel()
        _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
        return counts[first], inverse.ravel()
    unique, inverse = np.unique(counts, axis=0, return_inverse=True)
    return unique, inverse.ravel()


# Keys of `getAdvancedMeasures` in the order of its result dict; the markers stand for the values that
# are not plain array items.
_SIGN_T, _SIGN_F, _STEP, _BASIC = "signal_sign_T", "signal_sign_F", "timestep", "basic_measures"
_DICT_KEYS = (
    "TT", "TF", "FT", "FF", "TT_c", "TF_c", "FT_c", "FF_c", "conditionT", "conditionF", "targetT", "targetF",
    "isNegativeCorrelated", "isPositiveCorrelated", "supportTT", "supportFT", "supportTF", "supportFF",
    _SIGN_T, _SIGN_F, _STEP, "bestFitP", "bestFitN", "is_essential_gene", "Noise_P", "Noise_N", "Signal_P",
    "Signal_N", "pickT_support", "pickF_support", "pickT_causality_test", "pickF_causality_test",
    "pickT_confidenceCounter", "pickF_confidenceCounter", "pickT_all_confidence", "pickF_all_confidence",
    "pickT_max_confidence", "pickF_max_confidence", _BASIC, "p_value", "chiSQ_value", "pickT_mutualInfo",
    "pickF_mutualInfo",
)


class _LazyMeasures:
    """The `getAdvancedMeasures` dicts of the distinct count rows of one step, built on first access.

    Only the few candidates that survive the node filters ever need their dict, so the arrays returned
    by `getAdvancedMeasuresBatch` are turned into dicts on demand (and cached, so that candidates with
    the same counts share one dict, as before).
    """

    def __init__(self, arrays: Dict[str, np.ndarray], rows: np.ndarray, constants: Dict[str, int],
                 show_basic_measures: bool):
        self._arrays = arrays
        self._rows = rows
        self._constants = constants
        self._show_basic_measures = show_basic_measures
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        entry = self._cache.get(index)
        if entry is None:
            entry = self._cache[index] = self._build(index)
        return entry

    def _build(self, i: int) -> Dict[str, Any]:
        arrays = self._arrays
        entry: Dict[str, Any] = {}
        for key in _DICT_KEYS:
            if key == _SIGN_T:
                entry[key] = "TF" if arrays["signal_sign_T_is_TF"].item(i) else "TT"
            elif key == _SIGN_F:
                entry[key] = "FF" if arrays["signal_sign_F_is_FF"].item(i) else "FT"
            elif key == _STEP:
                entry[key] = self._constants["timestep"]
            elif key == _BASIC:
                if self._show_basic_measures:
                    entry[key] = _basic_measures(self._rows[i].tolist(), self._constants)
            else:
                entry[key] = arrays[key].item(i)
        return entry


def _step_measures(
    cache: Dict[str, Any],
    target_gene: str,
    candidate_genes: Sequence[str],
    fixed_state: Dict[str, int],
    step: int,
    show_basic_measures: bool,
) -> Dict[str, Any]:
    counted = count_arrays(cache, target_gene, candidate_genes, fixed_state, step)
    time_step = step + 1
    constants = {
        "target_T_count": counted["target_T_count"],
        "target_F_count": counted["target_F_count"],
        "total_calculated_timepoints": cache["n_timepoints"] - cache["total_samples"] * time_step,
        "num_of_conditional_genes": len(fixed_state) + 1,
        "timestep": time_step,
    }
    unique, inverse = _unique_rows(counted["counts"])
    arrays = _core.getAdvancedMeasuresBatch(
        np.ascontiguousarray(unique, dtype=np.int64),
        constants["target_T_count"], constants["target_F_count"],
        constants["total_calculated_timepoints"], time_step,
    )
    numeric = np.stack([
        arrays["bestFitP"], arrays["bestFitN"],
        np.full(len(unique), time_step, dtype=np.float64),
        arrays["is_essential_gene"], arrays["isPositiveCorrelated"], arrays["isNegativeCorrelated"],
    ], axis=1).astype(np.float64, copy=False)
    objects = _LazyMeasures(arrays, unique, constants, show_basic_measures)
    return {"objects": objects, "numeric": numeric, "inverse": inverse}


def _pick_best_steps(steps: List[Dict[str, Any]], n: int, temporal: int):
    """Per candidate, the temporal step with the best fit for the P and N rules, and whether it is essential."""
    rows = np.arange(n)
    # (steps, candidates, 6): bestFitP, bestFitN, timestep, essential, positively and negatively correlated
    table = np.stack([s["numeric"][s["inverse"]] for s in steps])
    inverse = np.stack([s["inverse"] for s in steps])  # (steps, candidates): distinct-row index per step

    # Best fit across temporal steps, replicating the sequential comparison of the original loop.
    pick_p = np.zeros(n, dtype=np.int64)
    pick_n = np.zeros(n, dtype=np.int64)
    for i in range(1, temporal):
        new_p, new_n, new_ts = table[i, :, 0], table[i, :, 1], table[i, :, 2]
        pick_p = np.where(table[pick_p, rows, 0] > new_p, i, pick_p)
        pick_n = np.where(table[pick_n, rows, 1] > new_n, i, pick_n)
        pick_p = np.where((table[pick_p, rows, 0] == new_p) & (table[pick_p, rows, 2] > new_ts), i, pick_p)
        pick_n = np.where((table[pick_n, rows, 1] == new_n) & (table[pick_n, rows, 2] > new_ts), i, pick_n)

    ess_p = table[pick_p, rows, 3] > 0
    ess_n = table[pick_n, rows, 3] > 0
    return table, inverse, pick_p, pick_n, ess_p, ess_n


def has_essential_candidate(
    cache: Dict[str, Any], target_gene: str, genes: Sequence[str], temporal: int
) -> bool:
    """True when some gene of `genes` shows an essential relationship with `target_gene`.

    Equivalent to ``_tree.filterTargetGenesByConditionGenes`` keeping the target, but evaluated for all
    candidates at once (that C++ filter tries the genes one by one and is quadratic in the gene count).
    """
    n = len(genes)
    if n == 0:
        return False
    steps = [_step_measures(cache, target_gene, genes, {}, s, False) for s in range(temporal)]
    *_, ess_p, ess_n = _pick_best_steps(steps, n, temporal)
    return bool((ess_p | ess_n).any())


def node_measurements(
    cache: Dict[str, Any],
    target_gene: str,
    genes: Sequence[str],
    fixed_state: Dict[str, int],
    temporal: int,
    show_basic_measures: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Columnar equivalent of `tensor_mining.gene_probabilities_measurements` (same result)."""
    n = len(genes)
    if n == 0:
        return {}
    steps = [_step_measures(cache, target_gene, genes, fixed_state, s, show_basic_measures) for s in range(temporal)]

    rows = np.arange(n)
    table, inverse, pick_p, pick_n, ess_p, ess_n = _pick_best_steps(steps, n, temporal)
    keep = ess_p | ess_n

    # The chosen dict of every candidate is identified by its temporal step.
    # "if p is not essential: p = n", then "if n is not essential: n = p" (with the new p)
    p_step = np.where(ess_p, pick_p, pick_n)
    n_step = np.where(ess_n, pick_n, p_step)
    # "if p has no correlation: p = n"
    p_uncorrelated = ~((table[p_step, rows, 4] > 0) | (table[p_step, rows, 5] > 0))
    p_step = np.where(p_uncorrelated, n_step, p_step)
    n_uncorrelated = ~((table[n_step, rows, 4] > 0) | (table[n_step, rows, 5] > 0))

    result: Dict[str, Dict[str, Any]] = {}
    objects = [s["objects"] for s in steps]
    for k in np.flatnonzero(keep).tolist():
        p = objects[p_step[k]][inverse[p_step[k], k]]
        other = objects[n_step[k]][inverse[n_step[k], k]]
        if n_uncorrelated[k]:
            # "if n has no correlation: skip when p == n, otherwise n = p"
            if p is other or p == other:
                continue
            other = p
        result[genes[k]] = {"probabilityOfFourCombines_P": p, "probabilityOfFourCombines_N": other}
    return result
