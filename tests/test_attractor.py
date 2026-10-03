import copy
import os
import unittest

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from fbnnet_core.attractor import (
    is_satisfied,
    get_probability_from_function_input,
    get_fbm_successor,
    network_fix_update,
    search_for_attractors,
    reconstruct_timeseries,
)
from fbnnet_core.boolnet import load_network
from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from fbnnet_core.general_utils import generate_similary_report
from fbnnet_core.network import mine_fbn_network
from fbnnet_core.network_graph import draw_attractor, plot_network
import numpy as np

EXAMPLE_BN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example.bn")


def _simple_network():
    """A -> A (self-sustaining once on), B has no regulators (always off)."""
    genes = ["A", "B"]
    return {
        "class": "FundamentalBooleanNetwork",
        "genes": genes,
        "interactions": {
            "A": {"A_1_Activator": {"input": [1], "expression": "A", "type": 1, "probability": 1.0, "timestep": 1}},
            "B": {},
        },
        "fixed": {"A": -1, "B": -1},
        "timedecay": {"A": 1, "B": 1},
    }


class TestIsSatisfied(unittest.TestCase):
    def test_literal_one_is_always_true(self):
        self.assertTrue(is_satisfied({"A": 0}, ["1"]))

    def test_literal_zero_is_always_false(self):
        self.assertFalse(is_satisfied({"A": 1}, ["0"]))

    def test_positive_reference_requires_gene_on(self):
        self.assertTrue(is_satisfied({"A": 1}, ["A"]))
        self.assertFalse(is_satisfied({"A": 0}, ["A"]))

    def test_negated_reference_requires_gene_off(self):
        self.assertTrue(is_satisfied({"A": 0}, ["!", "A"]))
        self.assertFalse(is_satisfied({"A": 1}, ["!", "A"]))


class TestGetProbabilityFromFunctionInput(unittest.TestCase):
    def test_matching_expression_returns_probability(self):
        prob = get_probability_from_function_input(1, "A", 0.8, {"A": 1})
        self.assertAlmostEqual(prob, 0.8)

    def test_non_matching_expression_returns_zero(self):
        prob = get_probability_from_function_input(1, "A", 0.8, {"A": 0})
        self.assertEqual(prob, 0.0)

    def test_invalid_func_type_raises(self):
        with self.assertRaises(ValueError):
            get_probability_from_function_input(2, "A", 0.8, {"A": 1})


class TestGetFBMSuccessor(unittest.TestCase):
    def test_self_activator_stays_on_once_on(self):
        network = _simple_network()
        genes = network["genes"]
        previous_states = np.array([[1], [0]])  # A=1, B=0 at step 1
        result = get_fbm_successor(network, previous_states, 2, genes, "synchronous")
        self.assertEqual(result["nextState"]["A"], 1)
        self.assertEqual(result["nextState"]["B"], 0)

    def test_fixed_gene_is_not_updated(self):
        network = _simple_network()
        network["fixed"]["A"] = 0  # fix A off regardless of its rule
        genes = network["genes"]
        previous_states = np.array([[1], [0]])
        result = get_fbm_successor(network, previous_states, 2, genes, "synchronous")
        self.assertEqual(result["nextState"]["A"], 1)  # fixed genes just carry forward
        self.assertEqual(result["decayIndex"]["A"], 0)


class TestNetworkFixUpdate(unittest.TestCase):
    def test_fixes_specified_genes(self):
        network = _simple_network()
        updated = network_fix_update(network, ["A"], [0])
        self.assertEqual(updated["fixed"]["A"], 0)

    def test_rejects_invalid_gene(self):
        network = _simple_network()
        with self.assertRaises(ValueError):
            network_fix_update(network, ["NotAGene"], [0])

    def test_rejects_invalid_value(self):
        network = _simple_network()
        with self.assertRaises(ValueError):
            network_fix_update(network, ["A"], [5])


class TestSearchForAttractors(unittest.TestCase):
    def test_finds_fixed_point_attractors(self):
        network = _simple_network()
        genes = network["genes"]
        result = search_for_attractors(network, genes, max_search=20)
        self.assertEqual(result["class"], "FBMAttractors")
        self.assertEqual(result["Genes"], genes)
        # Every state (A,B in {0,1}^2) should settle into an attractor.
        self.assertGreater(len(result["Attractors"]), 0)
        for attractor in result["Attractors"]:
            self.assertGreater(len(attractor), 0)

    def test_rejects_wrong_network_class(self):
        with self.assertRaises(ValueError):
            search_for_attractors({"class": "BooleanNetworkCollection", "genes": []}, [])

    def test_rejects_invalid_transition_type(self):
        network = _simple_network()
        with self.assertRaises(ValueError):
            search_for_attractors(network, network["genes"], transition_type="invalid")


class TestReconstructTimeseriesAndAttractorsVignette(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-attractor.R, using the
    ExampleNetwork (example.bn, 5 genes) end-to-end: mine a network from a
    generated training series, reconstruct the training series from the
    mined network, and verify the reconstruction matches perfectly - then
    search for attractors and draw one. Mirrors R's "run synchronous should
    succeed" describe block exactly, including its assertions on
    ErrorRate/AccurateRate/MissMatchedRate/PerfectMatchedRate.
    """

    @classmethod
    def setUpClass(cls):
        network = load_network(EXAMPLE_BN)
        genes = network["genes"]

        cls.initial_states = generateAllCombinationBinary(genes)
        cls.trainingseries = generateBoolNetTimeseries(
            network, cls.initial_states, 43, transition_type="synchronous"
        )

        cube = construct_fbn_cube(
            genes, genes,
            [pd.DataFrame(mat, index=genes, columns=[str(j + 1) for j in range(mat.shape[1])])
             for mat in cls.trainingseries],
            max_k=5, temporal=1, use_parallel=False,
        )
        cls.network = mine_fbn_network(cube, genes)

    def test_reconstruct_timeseries_matches_training_series(self):
        reconstructed = reconstruct_timeseries(
            self.network, self.initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=False,
        )

        report = generate_similary_report(self.trainingseries, reconstructed)
        self.assertEqual(report["ErrorRate"], 0)
        self.assertEqual(report["AccurateRate"], 1)
        self.assertEqual(report["MissMatchedRate"], 0)
        self.assertEqual(report["PerfectMatchedRate"], 1)

    def test_reconstruct_timeseries_parallel_matches_sequential(self):
        sequential = reconstruct_timeseries(
            self.network, self.initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=False,
        )
        parallel = reconstruct_timeseries(
            self.network, self.initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=True,
        )
        self.assertEqual(len(sequential), len(parallel))
        for seq_mat, par_mat in zip(sequential, parallel):
            np.testing.assert_array_equal(seq_mat, par_mat)

    def test_search_for_attractors_and_draw(self):
        attractor = search_for_attractors(
            self.network, genes=self.network["genes"],
            start_states=self.initial_states, transition_type="synchronous",
        )
        self.assertEqual(attractor["class"], "FBMAttractors")
        self.assertGreater(len(attractor["Attractors"]), 0)
        # Should be printable without error (R: expect_error(print(attractor), NA)).
        self.assertIsInstance(repr(attractor), str)

        # Display the dynamic trajectory of the second attractor (R's index 2 ->
        # Python's 0-based index 1), matching
        # FBNNetwork.Graph.DrawAttractor(network, attractor, 2).
        index = 1 if len(attractor["Attractors"]) > 1 else 0
        ax = draw_attractor(attractor, index=index)
        self.assertIsNotNone(ax)


class TestReconstructTimeseriesRParity(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-reconstructTimeseries.R on the
    ExampleNetwork (example.bn, maxK=5): re-mining from a reconstructed series
    (with decay 1 or 2, temporal 1 or 2) must reconstruct that series perfectly.
    """

    @classmethod
    def setUpClass(cls):
        network = load_network(EXAMPLE_BN)
        cls.genes = network["genes"]
        cls.initial_states = generateAllCombinationBinary(cls.genes)
        cls.trainingseries = generateBoolNetTimeseries(
            network, cls.initial_states, 43, transition_type="synchronous"
        )
        cls.network = cls._mine(cls.trainingseries, temporal=1)

    @classmethod
    def _mine(cls, series, temporal):
        frames = [
            pd.DataFrame(mat, index=cls.genes, columns=[str(j + 1) for j in range(mat.shape[1])])
            for mat in series
        ]
        cube = construct_fbn_cube(cls.genes, cls.genes, frames, max_k=5, temporal=temporal, use_parallel=False)
        return mine_fbn_network(cube, cls.genes)

    def _reconstruct(self, network):
        return reconstruct_timeseries(
            network, self.initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=False,
        )

    def _assert_perfect(self, expected, actual):
        report = generate_similary_report(expected, actual)
        self.assertEqual(report["ErrorRate"], 0)
        self.assertEqual(report["AccurateRate"], 1)
        self.assertEqual(report["MissMatchedRate"], 0)
        self.assertEqual(report["PerfectMatchedRate"], 1)

    def _decay_two_network(self):
        network = copy.deepcopy(self.network)
        for gene in self.genes[:5]:
            network["timedecay"][gene] = 2
        return network

    def test_first_reconstruction_matches_training_series(self):
        self._assert_perfect(self.trainingseries, self._reconstruct(self.network))

    def test_remined_network_is_identical_and_reconstructs_perfectly(self):
        reconstructed = self._reconstruct(self.network)
        remined = self._mine(reconstructed, temporal=1)
        plot_network(remined)

        self.assertEqual(list(remined["genes"]), list(self.network["genes"]))
        self.assertEqual(remined["interactions"], self.network["interactions"])
        self._assert_perfect(reconstructed, self._reconstruct(remined))

    def test_decay_two_reconstruction_roundtrip(self):
        reconstructed = self._reconstruct(self._decay_two_network())
        remined = self._mine(reconstructed, temporal=1)
        plot_network(remined)
        self._assert_perfect(reconstructed, self._reconstruct(remined))

    def test_temporal_two_reconstruction_roundtrip(self):
        reconstructed = self._reconstruct(self.network)
        remined = self._mine(reconstructed, temporal=2)
        plot_network(remined)
        self._assert_perfect(reconstructed, self._reconstruct(remined))

    def test_temporal_two_decay_two_reconstruction_roundtrip(self):
        reconstructed = self._reconstruct(self._decay_two_network())
        remined = self._mine(reconstructed, temporal=2)
        plot_network(remined)
        self._assert_perfect(reconstructed, self._reconstruct(remined))

    def test_get_probability_from_function_input_on_mined_network(self):
        interactions = self.network["interactions"][self.genes[0]]
        activator = next(f for f in interactions.values() if f["type"] == 1)
        pre_gene_inputs = {
            self.genes[index - 1]: self.initial_states[0][self.genes[index - 1]]
            for index in activator["input"]
        }
        probability = get_probability_from_function_input(
            1, activator["expression"], activator["probability"], pre_gene_inputs
        )
        self.assertIn(probability, (0.0, float(activator["probability"])))


if __name__ == "__main__":
    unittest.main()
