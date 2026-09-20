import unittest

import numpy as np
import pandas as pd

from py_src.application import (
    generate_fbm_network,
    binarize_time_series,
    is_boolean_type_timeseries_data,
)


def _build_boolean_training_series():
    from py_src.boolnet import load_network
    from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries

    with open("example.bn", "w") as f:
        f.write("targets, factors\n")
        f.write("Gene1, Gene1\n")
        f.write("Gene2, Gene1 & Gene5 & !Gene4\n")
        f.write("Gene3, Gene3\n")
        f.write("Gene4, Gene3 & !(Gene1 & Gene5)\n")
        f.write("Gene5, !Gene2\n")

    network = load_network("example.bn")
    initial_states = generateAllCombinationBinary(network["genes"])
    training_series = generateBoolNetTimeseries(network, initial_states, 43, transition_type="synchronous")
    training_series = [
        pd.DataFrame(mat, index=network["genes"], columns=[str(j + 1) for j in range(mat.shape[1])])
        for mat in training_series
    ]
    return network, training_series


class TestIsBooleanTypeTimeseriesData(unittest.TestCase):
    def test_boolean_data_returns_true(self):
        df = pd.DataFrame([[0, 1, 1], [1, 0, 0]], index=["A", "B"])
        self.assertTrue(is_boolean_type_timeseries_data([df]))

    def test_non_boolean_data_returns_false(self):
        df = pd.DataFrame([[0.1, 1.5, 2.0], [1, 0, 0]], index=["A", "B"])
        self.assertFalse(is_boolean_type_timeseries_data([df]))


class TestBinarizeTimeSeries(unittest.TestCase):
    def test_binarizes_numeric_series_into_two_clusters(self):
        df = pd.DataFrame([[0.0, 0.1, 9.8, 10.0]], index=["A"], columns=["1", "2", "3", "4"])
        result = binarize_time_series([df])[0]
        self.assertTrue(is_boolean_type_timeseries_data([result]))
        self.assertEqual(list(result.loc["A"]), [0, 0, 1, 1])

    def test_constant_series_becomes_all_zero(self):
        df = pd.DataFrame([[5.0, 5.0, 5.0]], index=["A"], columns=["1", "2", "3"])
        result = binarize_time_series([df])[0]
        self.assertEqual(list(result.loc["A"]), [0, 0, 0])

    def test_rejects_unsupported_method(self):
        df = pd.DataFrame([[0.0, 1.0]], index=["A"], columns=["1", "2"])
        with self.assertRaises(ValueError):
            binarize_time_series([df], method="edgeDetector")


class TestGenerateFbmNetwork(unittest.TestCase):
    def test_pipeline_mines_network_covering_all_genes(self):
        network, training_series = _build_boolean_training_series()
        result = generate_fbm_network(training_series, max_k=5, max_deep_temporal=1)
        self.assertTrue(set(network["genes"]).issubset(set(result["genes"])))
        self.assertIn("interactions", result)

    def test_accepts_single_dataframe(self):
        _, training_series = _build_boolean_training_series()
        result = generate_fbm_network(training_series[0], max_k=3)
        self.assertIn("genes", result)

    def test_network_only_false_returns_cube_and_network(self):
        _, training_series = _build_boolean_training_series()
        result = generate_fbm_network(training_series, max_k=3, network_only=False)
        self.assertIn("cube", result)
        self.assertIn("network", result)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            generate_fbm_network([])

    def test_rejects_invalid_confidence_threshold(self):
        _, training_series = _build_boolean_training_series()
        with self.assertRaises(ValueError):
            generate_fbm_network(training_series, threshold_confidence=2)


if __name__ == "__main__":
    unittest.main()
