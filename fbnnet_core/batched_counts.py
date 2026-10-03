"""
Phase 3 foundation (see .temp/parallel_improvement_plan.md section 6.2): a
vectorized NumPy reimplementation of _utils.matchCount, batched over
multiple pattern vectors sharing the same matrix.

matchCount(m, v) counts columns of `m` (rows x cols) where every row exactly
equals the corresponding entry of pattern `v` (one value per row) - i.e. one
cell of a contingency table. getBasicMeasures (src/fbn_core.cpp) calls
matchCount up to 12 times per tree node against the same `m`/`mc` matrices.
batched_match_counts computes counts for ALL patterns sharing the same `m` in
a single vectorized pass, instead of N separate matchCount calls each
rebuilding the row-wise diff from scratch.

This module does not change the mining engine's control flow (still
depth-first, one tree node at a time) - it is an exactly-validated drop-in
replacement for the innermost counting primitive, intended as the foundation
for a future breadth-first/batched backend (the rest of Phase 3).
"""
import numpy as np


def batched_match_counts(m: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    """
    Vectorized equivalent of calling _utils.matchCount(m, v) once per
    row of `patterns`.

    Args:
        m: (n_rows, n_cols) array - same semantics as matchCount's `m`.
        patterns: (n_patterns, n_rows) array, or a single (n_rows,) pattern
            (treated as n_patterns=1).

    Returns:
        (n_patterns,) int array of match counts, one per pattern - identical
        to [matchCount(m, patterns[i]) for i in range(n_patterns)], but
        computed in one batched pass instead of n_patterns separate calls.
    """
    m = np.asarray(m, dtype=np.float64)
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim == 1:
        patterns = patterns[None, :]

    if m.ndim != 2:
        raise ValueError("m must be a 2D (n_rows, n_cols) array")
    if patterns.ndim != 2 or patterns.shape[1] != m.shape[0]:
        raise ValueError("patterns must be (n_patterns, n_rows) with n_rows == m.shape[0]")

    # diff[p, r, c] = |m[r, c] - patterns[p, r]|; a column c matches pattern p
    # iff every row's diff is 0, i.e. the column's diff sum over rows is 0.
    diff = np.abs(m[None, :, :] - patterns[:, :, None])
    col_sums = diff.sum(axis=1)  # (n_patterns, n_cols)
    return np.count_nonzero(col_sums == 0, axis=1)
