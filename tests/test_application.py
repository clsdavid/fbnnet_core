import os
import unittest
import warnings

import numpy as np
import pandas as pd
import pyreadr

from py_src.application import (
    generate_fbm_network,
    binarize_time_series,
    is_boolean_type_timeseries_data,
)
from py_src.attractor import reconstruct_timeseries
from py_src.data_utils import generateAllCombinationBinary
from py_src.general_utils import generate_similary_report

YEAST_TIME_SERIES_RDA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "yeastTimeSeries.rda"
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
            binarize_time_series([df], method="scanStatistic")

    def test_edge_detector_first_edge_and_max_edge(self):
        df = pd.DataFrame([[0, 5, 6, 7, 20, 21]], index=["A"], columns=list("123456"))
        first = binarize_time_series([df], method="edgeDetector")[0]
        self.assertEqual(list(first.loc["A"]), [0, 1, 1, 1, 1, 1])
        maximum = binarize_time_series([df], method="edgeDetector", edge="maxEdge")[0]
        self.assertEqual(list(maximum.loc["A"]), [0, 0, 0, 0, 1, 1])

    def test_edge_detector_unsorted_constant_and_scaling(self):
        df = pd.DataFrame([[10, 1, 3, 2], [5, 5, 5, 5]], index=["A", "B"], columns=list("1234"))
        result = binarize_time_series([df], method="edgeDetector")[0]
        self.assertEqual(list(result.loc["A"]), [1, 0, 0, 0])
        self.assertEqual(list(result.loc["B"]), [0, 0, 0, 0])
        scaled = pd.DataFrame([[0, 1, 2, 3, 100]], index=["A"], columns=list("12345"))
        self.assertEqual(list(binarize_time_series([scaled], method="edgeDetector", scaling=0.1)[0].loc["A"]), [0, 0, 0, 0, 1])

    def test_edge_detector_thresholds_over_all_matrices_together(self):
        first = pd.DataFrame([[0, 1]], index=["A"], columns=["1", "2"])
        second = pd.DataFrame([[2, 30]], index=["A"], columns=["1", "2"])
        result = binarize_time_series([first, second], method="edgeDetector")
        self.assertEqual(list(result[0].loc["A"]), [0, 0])
        self.assertEqual(list(result[1].loc["A"]), [0, 1])

    def test_edge_detector_rejects_unknown_edge(self):
        df = pd.DataFrame([[0.0, 1.0, 2.0]], index=["A"], columns=["1", "2", "3"])
        with self.assertRaises(ValueError):
            binarize_time_series([df], method="edgeDetector", edge="lastEdge")


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


class TestGenerateFbmNetworkReconstructionRParity(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-constructFBNCube.R
    "test with the sub cube" describe block: mine a network via
    generate_fbm_network (R's generateFBMNetwork) with maxK=4, reconstruct
    the training series, and verify a perfect reconstruction.
    """

    def test_sub_cube_pipeline_reconstructs_training_series_perfectly(self):
        network, training_series = _build_boolean_training_series()
        mined_network = generate_fbm_network(
            training_series, max_k=4, max_deep_temporal=1, use_parallel=False,
        )

        initial_states = generateAllCombinationBinary(network["genes"])
        reconstructed = reconstruct_timeseries(
            mined_network, initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=False,
        )

        report = generate_similary_report(training_series, reconstructed)
        self.assertEqual(report["ErrorRate"], 0)
        self.assertEqual(report["AccurateRate"], 1)
        self.assertEqual(report["MissMatchedRate"], 0)
        self.assertEqual(report["PerfectMatchedRate"], 1)


class TestGenerateFbmNetworkRealYeastTimeSeries(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-application.R, using the
    real yeastTimeSeries dataset (data/yeastTimeSeries.rda, read via
    pyreadr) that R's test relies on. Only the "kmeans" discretisation
    method is currently ported (see .temp/r-python-test-parity-audit.md);
    R's "edgeDetector" method is a documented, still-unimplemented gap.
    """

    @classmethod
    def setUpClass(cls):
        result = pyreadr.read_r(YEAST_TIME_SERIES_RDA)
        cls.yeast_time_series = result["yeastTimeSeries"]

    def test_kmeans_network_only_true(self):
        network = generate_fbm_network(self.yeast_time_series, method="kmeans", max_k=4)
        self.assertIn("genes", network)
        self.assertIn("interactions", network)

    def test_kmeans_network_only_false_returns_cube_and_network(self):
        result = generate_fbm_network(
            self.yeast_time_series, method="kmeans", max_k=4, network_only=False,
        )
        self.assertIn("cube", result)
        self.assertIn("network", result)

    def test_kmeans_verbose_true(self):
        network = generate_fbm_network(self.yeast_time_series, method="kmeans", max_k=4, verbose=True)
        self.assertIn("genes", network)

    def test_default_method(self):
        network = generate_fbm_network(self.yeast_time_series)
        self.assertIn("genes", network)

    def test_edge_detector_network_only_true_and_false(self):
        network = generate_fbm_network(self.yeast_time_series, method="edgeDetector")
        self.assertIn("interactions", network)
        result = generate_fbm_network(self.yeast_time_series, method="edgeDetector", network_only=False)
        self.assertIn("cube", result)
        self.assertIn("network", result)

    def test_no_warnings_for_any_r_test_application_call(self):
        calls = (
            {}, {"verbose": True}, {"method": "kmeans"}, {"method": "edgeDetector"}, {"network_only": False},
        )
        for kwargs in calls:
            with self.subTest(kwargs=kwargs):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    generate_fbm_network(self.yeast_time_series, **kwargs)


if __name__ == "__main__":
    unittest.main()
