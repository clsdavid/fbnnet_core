import os
import unittest

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from py_src.attractor import reconstruct_timeseries, search_for_attractors
from py_src.boolnet import load_network
from py_src.cube import construct_fbn_cube
from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from py_src.general_utils import generate_similary_report
from py_src.network import mine_fbn_network
from py_src.network_app import load_fbn_network, convert_to_boolean_network_collection
from py_src.network_graph import (
    to_networkx_graph,
    to_networkx_graph_with_rules,
    draw_static_network,
    draw_static_network_slice,
    draw_dynamic_network,
    plot_network,
    draw_attractor,
)

EXAMPLE_FBN_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_fbn.csv")
EXAMPLE_BN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example.bn")


class TestToNetworkxGraph(unittest.TestCase):
    def setUp(self):
        self.network = load_fbn_network(EXAMPLE_FBN_CSV)

    def test_graph_has_one_node_per_gene(self):
        graph = to_networkx_graph(self.network)
        self.assertEqual(set(graph.nodes), set(self.network["genes"]))

    def test_graph_has_decay_self_loops_when_requested(self):
        graph = to_networkx_graph(self.network, show_decay=True)
        for gene in self.network["genes"]:
            self.assertTrue(graph.has_edge(gene, gene))

    def test_rejects_non_fbn_network(self):
        with self.assertRaises(ValueError):
            to_networkx_graph({"class": "SomethingElse", "genes": []})


class TestDrawStaticNetwork(unittest.TestCase):
    def setUp(self):
        self.network = load_fbn_network(EXAMPLE_FBN_CSV)

    def test_returns_axes(self):
        ax = draw_static_network(self.network)
        self.assertIsNotNone(ax)


class TestToNetworkxGraphWithRules(unittest.TestCase):
    def setUp(self):
        self.network = load_fbn_network(EXAMPLE_FBN_CSV)

    def test_gene_nodes_and_rule_nodes_both_present(self):
        graph = to_networkx_graph_with_rules(self.network)
        gene_nodes = {n for n, d in graph.nodes(data=True) if d["node_kind"] == "gene"}
        rule_nodes = {n for n, d in graph.nodes(data=True) if d["node_kind"] == "rule"}
        self.assertEqual(gene_nodes, set(self.network["genes"]))
        self.assertTrue(rule_nodes)

    def test_rule_node_sits_between_input_gene_and_target_gene(self):
        graph = to_networkx_graph_with_rules(self.network)
        for rule_node, data in graph.nodes(data=True):
            if data["node_kind"] != "rule":
                continue
            predecessors = set(graph.predecessors(rule_node))
            successors = set(graph.successors(rule_node))
            self.assertTrue(predecessors, f"{rule_node} has no input gene edges")
            self.assertTrue(successors, f"{rule_node} has no target gene edge")
            self.assertIn(data["label"][0], ("+", "-"))

    def test_rejects_non_fbn_network(self):
        with self.assertRaises(ValueError):
            to_networkx_graph_with_rules({"class": "SomethingElse", "genes": []})


class TestDrawStaticNetworkWithRuleNodes(unittest.TestCase):
    def test_returns_axes(self):
        network = load_fbn_network(EXAMPLE_FBN_CSV)
        ax = draw_static_network(network, show_rule_nodes=True)
        self.assertIsNotNone(ax)


class TestPlotNetwork(unittest.TestCase):
    def setUp(self):
        self.network = load_fbn_network(EXAMPLE_FBN_CSV)

    def test_static_without_target_genes(self):
        ax = plot_network(self.network, direction="static")
        self.assertIsNotNone(ax)

    def test_static_with_output_network_returns_tuple(self):
        target_gene = self.network["genes"][0]
        network, ax = plot_network(
            self.network, target_genes=[target_gene], direction="static", output_network=True
        )
        self.assertIn(target_gene, network["genes"])
        self.assertIsNotNone(ax)

    def test_forward_requires_target_genes(self):
        with self.assertRaises(ValueError):
            plot_network(self.network, direction="forward")

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            plot_network(self.network, direction="sideways")


class TestDrawAttractor(unittest.TestCase):
    def test_draws_first_attractor(self):
        network = load_fbn_network(EXAMPLE_FBN_CSV)
        network = convert_to_boolean_network_collection(network)
        network["class"] = "FundamentalBooleanNetwork"
        attractors = search_for_attractors(network, network["genes"], max_search=20)
        self.assertGreater(len(attractors["Attractors"]), 0)
        ax = draw_attractor(attractors, index=0)
        self.assertIsNotNone(ax)

    def test_rejects_non_attractor_result(self):
        with self.assertRaises(ValueError):
            draw_attractor({"class": "SomethingElse"})

    def test_rejects_out_of_range_index(self):
        network = load_fbn_network(EXAMPLE_FBN_CSV)
        network["class"] = "FundamentalBooleanNetwork"
        attractors = search_for_attractors(network, network["genes"], max_search=20)
        with self.assertRaises(ValueError):
            draw_attractor(attractors, index=len(attractors["Attractors"]) + 5)


class TestFbnGraphicRParity(unittest.TestCase):
    """
    Python equivalent of R's tests/testthat/test-fbngraphic.R "run
    synchronous should succeed" describe block, using the same
    ExampleNetwork fixture (example.bn, mined with maxK=5, temporal=1).
    R's test only asserts "no error" (`expect_error(..., NA)`) for the
    static/staticSlice/dynamic graphing calls plus the attractor drawing,
    aside from the exact reconstruction-accuracy assertions, so this mirrors
    that same coverage.
    """

    @classmethod
    def setUpClass(cls):
        network = load_network(EXAMPLE_BN)
        genes = network["genes"]
        cls.initial_states = generateAllCombinationBinary(genes)
        raw_trainingseries = generateBoolNetTimeseries(
            network, cls.initial_states, 43, transition_type="synchronous"
        )
        cls.trainingseries = [
            pd.DataFrame(mat, index=genes, columns=[str(j + 1) for j in range(mat.shape[1])])
            for mat in raw_trainingseries
        ]
        cube = construct_fbn_cube(genes, genes, cls.trainingseries, max_k=5, temporal=1, use_parallel=False)
        cls.network = mine_fbn_network(cube, genes)

    def test_reconstruct_timeseries_matches_training_series_exactly(self):
        resultfile = reconstruct_timeseries(
            self.network, self.initial_states, transition_type="synchronous",
            max_timepoints=43, use_parallel=False,
        )
        report = generate_similary_report(self.trainingseries, resultfile)
        self.assertEqual(report["ErrorRate"], 0)
        self.assertEqual(report["AccurateRate"], 1)
        self.assertEqual(report["MissMatchedRate"], 0)
        self.assertEqual(report["PerfectMatchedRate"], 1)

    def test_static_graph_no_error(self):
        self.assertIsNotNone(draw_static_network(self.network))
        self.assertIsNotNone(plot_network(self.network, direction="static"))

    def test_static_slice_no_error(self):
        ax = plot_network(
            self.network, direction="staticSlice",
            timeseries_matrix=self.trainingseries[2], target_time_point=3,
        )
        self.assertIsNotNone(ax)
        self.assertIsNotNone(draw_static_network_slice(self.network, self.trainingseries[2], time_point=3))

    def test_dynamic_no_error(self):
        axes = plot_network(
            self.network, direction="dynamic",
            timeseries_matrix=self.trainingseries[4], start_time_point=1, end_time_point=5,
        )
        self.assertEqual(len(axes), 5)
        self.assertIsNotNone(draw_dynamic_network(self.network, self.trainingseries[4], from_time_point=1, to_time_point=5))

    def test_draw_attractor_no_error(self):
        attractor = search_for_attractors(self.network, self.network["genes"], self.initial_states)
        self.assertIsNotNone(draw_attractor(attractor, index=1))


if __name__ == "__main__":
    unittest.main()
