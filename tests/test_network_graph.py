import os
import unittest

import matplotlib
matplotlib.use("Agg")

from py_src.network_app import load_fbn_network, convert_to_boolean_network_collection
from py_src.attractor import search_for_attractors
from py_src.network_graph import (
    to_networkx_graph,
    to_networkx_graph_with_rules,
    draw_static_network,
    plot_network,
    draw_attractor,
)

EXAMPLE_FBN_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_fbn.csv")


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


if __name__ == "__main__":
    unittest.main()
