import os
import unittest

import pandas as pd

from py_src.boolnet import load_network
from py_src.cube import construct_fbn_cube
from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from py_src.network import mine_fbn_network
from py_src.network_app import (
    load_fbn_network,
    match_names,
    convert_to_boolean_network_collection,
    merge_network,
    filter_network_connections,
    filter_network_connections_by_genes,
    filter_network_connections_by_input_genes,
    find_all_input_genes,
    find_all_target_genes,
    find_all_backward_related_genes,
    find_backward_related_network_by_genes,
    find_all_forward_related_genes,
    find_forward_related_network_by_genes,
)

EXAMPLE_FBN_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_fbn.csv")
EXAMPLE_BN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example.bn")


class TestMatchNames(unittest.TestCase):
    def test_extracts_gene_names_from_expression(self):
        self.assertEqual(match_names("Gene2 & Gene3"), ["Gene2", "Gene3"])
        self.assertEqual(match_names("!Gene1 | Gene3"), ["Gene1", "Gene3"])

    def test_strips_boolean_operators(self):
        self.assertEqual(match_names("any(Gene1, Gene2)"), ["Gene1", "Gene2"])


class TestNetworkApp(unittest.TestCase):
    def setUp(self):
        self.network = load_fbn_network(EXAMPLE_FBN_CSV)

    def test_load_fbn_network_populates_inputs(self):
        # Every interaction should have resolved (non-empty) gene input indices.
        for gene, interactions in self.network["interactions"].items():
            for interaction in interactions:
                self.assertTrue(interaction["input"], f"{gene} has empty input list")

    def test_convert_to_boolean_network_collection(self):
        bnc = convert_to_boolean_network_collection(self.network)
        self.assertEqual(set(bnc["genes"]), set(self.network["genes"]))
        self.assertEqual(bnc["class"], "BooleanNetworkCollection")

    def test_merge_network_and_filter(self):
        bnc = convert_to_boolean_network_collection(self.network)
        merged = merge_network(bnc, bnc)
        filtered = filter_network_connections(merged)
        self.assertEqual(set(filtered["genes"]), set(self.network["genes"]))
        # Every gene regulates or is regulated by something in this example network.
        for gene, interactions in filtered["interactions"].items():
            self.assertTrue(interactions, f"{gene} should have surviving interactions")

    def _build_filtered_network(self):
        bnc = convert_to_boolean_network_collection(self.network)
        return filter_network_connections(merge_network(bnc, bnc))

    def test_find_all_input_and_target_genes(self):
        filtered = self._build_filtered_network()
        input_genes = find_all_input_genes(filtered["interactions"], filtered["genes"])
        target_genes = find_all_target_genes(filtered["interactions"])
        self.assertEqual(set(input_genes), {"Gene1", "Gene2", "Gene3"})
        self.assertEqual(set(target_genes), {"Gene1", "Gene2", "Gene3"})

    def test_filter_network_connections_by_genes(self):
        filtered = self._build_filtered_network()
        only_gene1 = filter_network_connections_by_genes(filtered, ["Gene1"], exclusive=False)
        self.assertIn("Gene1", only_gene1["interactions"])
        self.assertNotIn("Gene2", only_gene1["interactions"])
        self.assertNotIn("Gene3", only_gene1["interactions"])

    def test_filter_network_connections_by_input_genes(self):
        filtered = self._build_filtered_network()
        by_input = filter_network_connections_by_input_genes(filtered, ["Gene1"])
        self.assertIn("Gene1", by_input["genes"])

    def test_filter_network_connections_by_input_genes_raises_on_empty_list(self):
        filtered = self._build_filtered_network()
        with self.assertRaises(ValueError):
            filter_network_connections_by_input_genes(filtered, [])

    def test_find_all_backward_related_genes(self):
        filtered = self._build_filtered_network()
        # Gene1 <- Gene2 & Gene3; Gene2 <- !Gene1 | Gene3; Gene3 <- Gene1 (cycle covers all genes)
        backward = find_all_backward_related_genes(filtered, "Gene1", max_deep=2)
        self.assertEqual(set(backward["genes"]), {"Gene1", "Gene2", "Gene3"})

    def test_find_backward_related_network_by_genes(self):
        filtered = self._build_filtered_network()
        backward = find_backward_related_network_by_genes(filtered, ["Gene1"], max_deep=2)
        self.assertIn("Gene1", backward["genes"])

    def test_find_backward_related_network_by_genes_raises_on_empty_list(self):
        filtered = self._build_filtered_network()
        with self.assertRaises(ValueError):
            find_backward_related_network_by_genes(filtered, [])

    def test_find_all_forward_related_genes(self):
        filtered = self._build_filtered_network()
        forward = find_all_forward_related_genes(filtered, "Gene1", max_deep=2)
        # Gene1 is an input to Gene1 (via Gene2&Gene3->Gene1 cycle) and Gene3 (Gene1->Gene3).
        self.assertIn("Gene3", forward["interactions"])
        self.assertTrue(forward["interactions"]["Gene3"])

    def test_find_forward_related_network_by_genes(self):
        filtered = self._build_filtered_network()
        forward = find_forward_related_network_by_genes(filtered, ["Gene1"], max_deep=2)
        self.assertIsNotNone(forward)

    def test_find_forward_related_network_by_genes_raises_on_empty_list(self):
        filtered = self._build_filtered_network()
        with self.assertRaises(ValueError):
            find_forward_related_network_by_genes(filtered, [])


class TestNetworkApplicationRParity(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-networkapplication.R, using
    the same ExampleNetwork fixture (example.bn, mined with maxK=5,
    temporal=1) that R's test relies on.
    """

    @classmethod
    def setUpClass(cls):
        network = load_network(EXAMPLE_BN)
        genes = network["genes"]
        initial_states = generateAllCombinationBinary(genes)
        trainingseries = generateBoolNetTimeseries(network, initial_states, 43, transition_type="synchronous")
        cube = construct_fbn_cube(
            genes, genes,
            [pd.DataFrame(mat, index=genes, columns=[str(j + 1) for j in range(mat.shape[1])])
             for mat in trainingseries],
            max_k=5, temporal=1, use_parallel=False,
        )
        cls.network = mine_fbn_network(cube, genes)

    def test_filter_by_genes_target_gene1_exclusive_true_expand_false(self):
        tt = filter_network_connections_by_genes(self.network, ["Gene1"], exclusive=True, expand=False)
        self.assertNotIn("Gene1", tt["interactions"])

    def test_filter_by_genes_target_gene1_exclusive_false_expand_false(self):
        tt = filter_network_connections_by_genes(self.network, ["Gene1"], exclusive=False, expand=False)
        self.assertEqual(len(tt["interactions"]), 1)

    def test_forward_related_target_gene1_regulation_0_target_0_deep_1(self):
        tt = find_forward_related_network_by_genes(self.network, ["Gene1"], 0, 0, 1)
        self.assertEqual(len(tt["interactions"]), 2)
        self.assertEqual(len(tt["interactions"]["Gene1"]), 1)
        self.assertEqual(len(tt["interactions"]["Gene2"]), 1)
        self.assertEqual(tt["interactions"]["Gene1"]["Gene1_1_Inhibitor"]["expression"], "!Gene1")
        self.assertEqual(tt["interactions"]["Gene2"]["Gene2_1_Inhibitor"]["expression"], "!Gene1")

    def test_forward_related_target_gene1_regulation_0_target_1_deep_1(self):
        tt = find_forward_related_network_by_genes(self.network, ["Gene1"], 0, 1, 1)
        self.assertEqual(len(tt["interactions"]), 3)
        self.assertEqual(tt["interactions"]["Gene4"]["Gene4_1_Inhibitor"]["expression"], "Gene1&Gene5")


if __name__ == "__main__":
    unittest.main()
