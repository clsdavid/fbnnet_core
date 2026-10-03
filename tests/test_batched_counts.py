"""
Validates py_src.batched_counts.batched_match_counts (Phase 3 foundation
kernel, see .temp/parallel_improvement_plan.md section 6.2) produces results
exactly identical to looping fbnnet_utils.matchCount per pattern - the
existing, already-correct C++ primitive. This is a prerequisite correctness
check before any future breadth-first/batched mining engine work can build
on top of this kernel.
"""
import unittest

import numpy as np
import fbnnet_utils

from py_src.batched_counts import batched_match_counts


def _reference_counts(m, patterns):
    """Loop over fbnnet_utils.matchCount, one call per pattern (ground truth)."""
    patterns = np.atleast_2d(patterns)
    return np.array([fbnnet_utils.matchCount(m, pattern) for pattern in patterns])


class TestBatchedMatchCounts(unittest.TestCase):
    def test_matches_single_pattern_fixture_from_test_fbn_utility(self):
        # Same fixture as tests/test_fbn_utility.py::test_matchCount.
        m = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        pattern = np.array([0, 1, 1, 1], dtype=np.float64)
        expected = fbnnet_utils.matchCount(m, pattern)
        result = batched_match_counts(m, pattern)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(int(result[0]), expected)

    def test_matches_getbasicmeasures_style_2x2_patterns(self):
        # Same shape of call getBasicMeasures makes: a 2-row m (conditional
        # gene + target gene states) against the four 2x2 contingency cells.
        state_matrix = np.array(
            [[0, 1, 0, 1, 9, 0, 1, 0, 1, 9, 1, 1, 0],
             [1, 1, 1, 9, 1, 1, 0, 1, 9, 1, 0, 0, 1]],
            dtype=np.float64,
        )
        patterns = np.array([
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        expected = _reference_counts(state_matrix, patterns)
        result = batched_match_counts(state_matrix, patterns)
        np.testing.assert_array_equal(result, expected)

    def test_matches_reference_on_random_matrices(self):
        rng = np.random.default_rng(42)
        for n_rows, n_cols, n_patterns in [(1, 1, 1), (2, 10, 4), (3, 50, 6), (5, 200, 8)]:
            with self.subTest(n_rows=n_rows, n_cols=n_cols, n_patterns=n_patterns):
                m = rng.integers(0, 2, size=(n_rows, n_cols)).astype(np.float64)
                patterns = rng.integers(0, 2, size=(n_patterns, n_rows)).astype(np.float64)
                expected = _reference_counts(m, patterns)
                result = batched_match_counts(m, patterns)
                np.testing.assert_array_equal(result, expected)

    def test_rejects_mismatched_pattern_length(self):
        m = np.zeros((3, 5))
        bad_patterns = np.zeros((2, 4))  # wrong row count (4 != 3)
        with self.assertRaises(ValueError):
            batched_match_counts(m, bad_patterns)


if __name__ == "__main__":
    unittest.main()
