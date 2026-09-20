import unittest

from py_src.attractor import (
    is_satisfied,
    get_probability_from_function_input,
    get_fbm_successor,
    network_fix_update,
    search_for_attractors,
)
import numpy as np


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


if __name__ == "__main__":
    unittest.main()
