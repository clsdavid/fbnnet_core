import os
import unittest

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from fbnnet_core.attractor import reconstruct_timeseries, search_for_attractors
from fbnnet_core.boolnet import load_network
from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from fbnnet_core.general_utils import generate_similary_report
from fbnnet_core.network import mine_fbn_network
from fbnnet_core.network_app import load_fbn_network, convert_to_boolean_network_collection
from fbnnet_core.datasets import load_dataset
from fbnnet_core.fbn_types import FundamentalBooleanNetwork
from fbnnet_core.network_graph import (
    FBNGraph,
    attractor_to_matrix,
    convert_to_network_graphic_object,
    convert_to_ngo,
    draw_attractor,
    draw_dynamic_for_one_matrix,
    fbn_network_graph,
    plot_attractor_heatmap,
    plot_network,
    static_network_in_slice,
    to_networkx_graph,
    to_networkx_graph_with_rules,
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


def _tiny_network():
    """A -> B with a single activator rule B_1_Activator = A (probability 1, no decay)."""
    return FundamentalBooleanNetwork({
        "genes": ["A", "B"],
        "interactions": {"B": {"B_1_Activator": {
            "input": [1], "expression": "A", "error": 0.0, "type": 1.0,
            "probability": 1.0, "support": 0.5, "timestep": 1.0}}},
        "fixed": {"A": -1, "B": -1},
        "timedecay": {"A": 0, "B": 0},
        "class": "FundamentalBooleanNetwork",
    })


def _series(a_values):
    return pd.DataFrame([a_values, [0] * len(a_values)], index=["A", "B"], columns=range(1, len(a_values) + 1))


class TestConvertToNetworkGraphicObject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_dataset("FBNExampleNetworks")
        cls.obj = convert_to_network_graphic_object(cls.network)

    def test_node_and_edge_columns_match_r(self):
        self.assertEqual(list(self.obj["nodes"].columns),
                         ["id", "shape", "color", "type", "value", "label", "shadow", "group", "title"])
        self.assertEqual(list(self.obj["edges"].columns), [
            "from", "to", "support", "targetNode", "title", "arrowtail", "arrowhead", "color", "lty",
            "arrow.mode", "arrow.size", "type", "arrows", "label", "dashes", "shadow"])

    def test_genes_and_rules_are_nodes_with_r_styling(self):
        nodes = self.obj["nodes"]
        genes = nodes[nodes["type"] == "gene"]
        rules = nodes[nodes["type"] == "TF"]
        self.assertEqual(list(genes["id"]), self.network["genes"])
        self.assertTrue((genes["color"] == "lightblue").all())
        self.assertTrue((genes["shape"] == "ellipse").all())
        n_rules = sum(len(v) for v in self.network["interactions"].values())
        self.assertEqual(len(rules), n_rules)
        self.assertTrue((rules["shape"] == "box").all())
        activators = rules[rules["id"].str.endswith("Activator")]
        inhibitors = rules[rules["id"].str.endswith("Inhibitor")]
        self.assertTrue((activators["color"] == "LightGreen").all() and activators["label"].str.startswith("+, ").all())
        self.assertTrue((inhibitors["color"] == "orange").all() and inhibitors["label"].str.startswith("-, ").all())

    def test_edges_one_rule_edge_plus_one_per_input(self):
        edges = self.obj["edges"]
        n_edges = sum(1 + len(r["input"]) for rules in self.network["interactions"].values() for r in rules.values())
        self.assertEqual(len(edges), n_edges)
        self.assertEqual(list(edges.index), list(range(1, n_edges + 1)))
        self.assertEqual((edges["type"] == "TF_to_Gene").sum(), sum(len(v) for v in self.network["interactions"].values()))

    def test_negated_input_is_red_dashed_and_plain_input_green(self):
        edges = self.obj["edges"]
        # Gene2_1_Activator: Gene1&!Gene4&Gene5
        rule = edges[(edges["to"] == "Gene2_1_Activator") & (edges["type"] == "Gene_to_TF")].set_index("from")
        self.assertEqual(rule.loc["Gene4", "color"], "red")
        self.assertTrue(rule.loc["Gene4", "dashes"])
        self.assertEqual(rule.loc["Gene1", "color"], "green")
        self.assertFalse(rule.loc["Gene1", "dashes"])

    def test_rule_edges_follow_regulation_type(self):
        edges = self.obj["edges"]
        activate = edges[edges["from"] == "Gene2_1_Activator"].iloc[0]
        inhibit = edges[edges["from"] == "Gene2_1_Inhibitor"].iloc[0]
        self.assertEqual((activate["color"], activate["dashes"], activate["arrows"]), ("darkblue", False, "to"))
        self.assertEqual((inhibit["color"], inhibit["dashes"], inhibit["arrows"]), ("darkred", True, "to"))

    def test_decay_edges_only_when_requested(self):
        self.assertFalse((self.obj["edges"]["type"] == "decay").any())
        with_decay = convert_to_network_graphic_object(self.network, show_decay=True)
        decay = with_decay["edges"][with_decay["edges"]["type"] == "decay"]
        self.assertEqual(sorted(decay["from"]), sorted(self.network["interactions"]))
        self.assertTrue((decay["from"] == decay["to"]).all())

    def test_gene_tooltips_use_david_annotation(self):
        net = load_dataset("FBM_Leukeamia_Networks")
        david = load_dataset("DAVID_Gene_List")
        obj = convert_to_network_graphic_object(net, david)
        titles = dict(zip(obj["nodes"]["id"], obj["nodes"]["title"]))
        self.assertEqual(titles["CDC42EP3"], "CDC42 effector protein 3(CDC42EP3)")
        self.assertEqual(convert_to_network_graphic_object(net)["nodes"].iloc[0]["title"], net["genes"][0])

    def test_rejects_non_fbn_network(self):
        with self.assertRaises(ValueError):
            convert_to_network_graphic_object({"class": "x", "genes": []})


class TestConvertToNgo(unittest.TestCase):
    def test_consecutive_activations_are_united(self):
        net = _tiny_network()
        dyn = convert_to_ngo(_series([1, 1, 0, 1, 1, 0]), net, convert_to_network_graphic_object(net))
        rule_rows = dyn[dyn["functype"] == "activate"]
        # the rule fired at steps 2, 3, 5 and 6 -> (1,3) and (4,6), for both of its edges (rule->gene, gene->rule)
        self.assertEqual(sorted(set(zip(rule_rows["onset"], rule_rows["terminus"]))), [(1, 3), (4, 6)])
        self.assertEqual(len(rule_rows), 4)
        self.assertEqual(set(rule_rows["duration"]), {2})
        self.assertEqual(list(dyn.columns), [
            "onset", "terminus", "tail", "head", "onset.censored", "terminus.censored",
            "duration", "edge.id", "fromState", "functype"])

    def test_trailing_isolated_interval_is_not_reported_like_r(self):
        net = _tiny_network()
        dyn = convert_to_ngo(_series([1, 1, 0, 1, 0]), net, convert_to_network_graphic_object(net))
        self.assertEqual(set(zip(dyn["onset"], dyn["terminus"])), {(1, 3)})

    def test_decay_interval_recorded_when_nothing_fires(self):
        net = _tiny_network()
        net["timedecay"] = {"A": 1, "B": 1}
        obj = convert_to_network_graphic_object(net, show_decay=True)
        dyn = convert_to_ngo(_series([0, 0, 0, 0]), net, obj)
        self.assertTrue((dyn["functype"] == "decay").any())

    def test_requires_dataframe(self):
        net = _tiny_network()
        with self.assertRaises(ValueError):
            convert_to_ngo([[1, 0]], net, convert_to_network_graphic_object(net))


class TestStaticSliceAndDynamicGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_dataset("FBNExampleNetworks")
        cls.series = load_dataset("ExampleTimeseriesData")[0]

    def test_static_graph_has_legend_and_title(self):
        graph = fbn_network_graph(self.network)
        self.assertEqual(graph.title, "Fundamental Boolean Networks")
        self.assertEqual(list(graph.legend_nodes["label"]),
                         ["Gene", "Activate Function (+, Timestep)", "Inhibit Function (-, Timestep)"])
        self.assertEqual(list(graph.legend_edges["label"]), ["activate", "inhibit", "activated input", "deactivated input"])

    def test_slice_contains_only_active_rules_and_all_genes(self):
        net = _tiny_network()
        obj = convert_to_network_graphic_object(net)
        dyn = convert_to_ngo(_series([1, 1, 0, 1, 1, 0]), net, obj)
        at_2 = static_network_in_slice(obj, 2, dyn)
        at_4 = static_network_in_slice(obj, 4, dyn)
        at_5 = static_network_in_slice(obj, 5, dyn)
        at_7 = static_network_in_slice(obj, 7, dyn)
        self.assertEqual(len(at_2.edges), 2)
        self.assertEqual(len(at_4.edges), 0)
        self.assertEqual(len(at_5.edges), 2)
        self.assertEqual(set(at_7.nodes["id"]), {"A", "B"})
        self.assertIn("B_1_Activator", set(at_2.nodes["id"]))
        self.assertTrue(at_2.title.endswith("2"))

    def test_dynamic_graph_colours_genes_by_state_and_numbers_nodes(self):
        net = _tiny_network()
        obj = convert_to_network_graphic_object(net, show_decay=True)
        series = _series([1, 1, 0, 0])
        graph = fbn_network_graph(net, type="dynamic", timeseries_matrix=series, from_time_point=1, to_time_point=3,
                                  network_object=obj)
        nodes = graph.nodes.set_index("id")
        self.assertEqual(nodes.loc["A_1", "color"], "lightblue")
        self.assertEqual(nodes.loc["A_2", "color"], "lightblue")
        self.assertEqual(nodes.loc["B_2", "color"], "pink")
        self.assertEqual(nodes.loc["B_1_Activator_2", "type"], "TF") if "B_1_Activator_2" in nodes.index else None
        self.assertTrue(graph.title.endswith("from the time point of 1 to 3"))
        self.assertEqual(list(graph.legend_nodes["label"])[:2], ["Activated Gene", "Inhibited Gene"])
        self.assertIn("decay", list(graph.legend_edges["label"]))

    def test_dynamic_graph_levels_increase_left_to_right(self):
        graph = fbn_network_graph(self.network, type="dynamic", timeseries_matrix=self.series, to_time_point=4)
        levels = graph.levels
        self.assertEqual(levels["Gene1_1"], 1)
        self.assertGreater(levels["Gene1_4"], levels["Gene1_2"])
        self.assertTrue(all(node_id in levels for node_id in graph.nodes["id"]))

    def test_slice_and_dynamic_require_timeseries(self):
        with self.assertRaises(ValueError):
            fbn_network_graph(self.network, type="dynamic")
        with self.assertRaises(ValueError):
            fbn_network_graph(self.network, type="staticSlice")

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            fbn_network_graph(self.network, type="sideways")

    def test_time_points_are_validated(self):
        with self.assertRaises(ValueError):
            fbn_network_graph(self.network, type="dynamic", timeseries_matrix=self.series, to_time_point=500)

    def test_network_without_interactions_gives_none(self):
        empty = FundamentalBooleanNetwork({"genes": ["A"], "interactions": {}, "class": "FundamentalBooleanNetwork"})
        self.assertIsNone(fbn_network_graph(empty))


class TestAttractorGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_dataset("FBNExampleNetworks")
        cls.attractors = search_for_attractors(cls.network, cls.network["genes"])

    def test_attractor_matrix(self):
        longest = max(range(len(self.attractors["Attractors"])), key=lambda i: len(self.attractors["Attractors"][i]))
        matrix = attractor_to_matrix(self.attractors, longest)
        self.assertEqual(list(matrix.index), self.network["genes"])
        self.assertEqual(list(matrix.columns), list(range(1, len(self.attractors["Attractors"][longest]) + 1)))

    def test_attractor_graph_is_layered_and_contains_every_gene_at_every_step(self):
        longest = max(range(len(self.attractors["Attractors"])), key=lambda i: len(self.attractors["Attractors"][i]))
        length = len(self.attractors["Attractors"][longest])
        graph = draw_attractor(self.network, self.attractors, longest)
        self.assertTrue(graph.hierarchical)
        self.assertIn("level", graph.nodes.columns)
        genes = graph.nodes[graph.nodes["type"] == "gene"]
        self.assertEqual(len(genes), len(self.network["genes"]) * length)
        self.assertEqual(graph.title, f"Dynamic Fundamental Boolean Networks from the time point of 1 to {length}")
        self.assertIsNotNone(graph.draw())

    def test_dynamic_for_one_matrix(self):
        graph = draw_dynamic_for_one_matrix(self.network, load_dataset("ExampleTimeseriesData")[0].iloc[:, :4])
        self.assertTrue(graph.hierarchical)

    def test_heatmap(self):
        self.assertIsNotNone(plot_attractor_heatmap(self.attractors, 0))

    def test_rejects_non_attractor_result(self):
        with self.assertRaises(ValueError):
            draw_attractor(self.network, {"class": "SomethingElse"})

    def test_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            draw_attractor(self.network, self.attractors, index=len(self.attractors["Attractors"]) + 5)


class TestPlotNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_dataset("FBM_Leukeamia_Networks")

    def test_static_without_target_genes(self):
        small = load_dataset("FBNExampleNetworks")
        self.assertIsInstance(plot_network(small), FBNGraph)

    def test_static_with_output_network_returns_tuple(self):
        network, graph = plot_network(self.network, target_genes=["CDC42EP3"], type="static", output_network=True)
        self.assertIn("CDC42EP3", network["genes"])
        self.assertIsInstance(graph, FBNGraph)

    def test_forward_and_backward_types(self):
        for plot_type in ("forward_1a", "forward_2a", "forward_3a", "forward_4a", "forward_1b", "forward_2b",
                          "forward_3b", "forward_4b", "backward_1a", "backward_2a", "backward_1b", "backward_2b"):
            with self.subTest(plot_type=plot_type):
                network = plot_network(self.network, ["CDC42EP3"], type=plot_type, output_network=True)[0]
                self.assertIsInstance(network, dict)

    def test_backward_graph_contains_upstream_regulators(self):
        network, graph = plot_network(self.network, ["CDC42EP3"], type="backward_1a", output_network=True)
        self.assertIn("CDC42EP3", network["interactions"])
        edges = graph.edges
        inputs = set(edges[(edges["type"] == "Gene_to_TF") & (edges["targetNode"] == "CDC42EP3")]["from"])
        self.assertTrue(inputs)
        self.assertTrue(all(edges[edges["targetNode"] == "CDC42EP3"]["title"].str.endswith("Inhibitor")))

    def test_expand_level_extends_forward_network(self):
        shallow = plot_network(self.network, ["CDC42EP3"], type="forward_1b", expand_level=1, output_network=True)[0]
        deep = plot_network(self.network, ["CDC42EP3"], type="forward_1b", expand_level=3, output_network=True)[0]
        self.assertGreaterEqual(len(deep["genes"]), len(shallow["genes"]))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            plot_network(self.network, ["CDC42EP3"], type="sideways")
        with self.assertRaises(ValueError):
            plot_network(self.network, ["CDC42EP3"], type="forward_9a")

    def test_rejects_non_fbn_network(self):
        with self.assertRaises(ValueError):
            plot_network({"class": "x"}, type="static")


class TestFBNGraphRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = fbn_network_graph(load_dataset("FBNExampleNetworks"))

    def test_draw_returns_axes_with_title(self):
        ax = self.graph.draw()
        self.assertEqual(ax.get_title(), "Fundamental Boolean Networks")

    def test_save_png_and_html(self):
        png = self.graph.save("network.png")
        html_path = self.graph.save("network.html")
        self.assertGreater(os.path.getsize(png), 1000)
        content = open(html_path, encoding="utf-8").read()
        self.assertIn("vis-network", content)
        self.assertIn("Fundamental Boolean Networks", content)
        self.assertIn("Gene1_1_Activator", content)

    def test_html_embeds_valid_json_and_hierarchical_option(self):
        import json
        import re

        page = self.graph.to_html()
        data = json.loads(re.search(r"var d = (.*);\n", page).group(1))
        self.assertEqual(len(data["nodes"]), len(self.graph.nodes))
        self.assertEqual(len(data["edges"]), len(self.graph.edges))
        self.assertNotIn("layout", data["options"])

        net = load_dataset("FBNExampleNetworks")
        attractors = search_for_attractors(net, net["genes"])
        longest = max(range(len(attractors["Attractors"])), key=lambda i: len(attractors["Attractors"][i]))
        page = draw_attractor(net, attractors, longest).to_html()
        data = json.loads(re.search(r"var d = (.*);\n", page).group(1))
        self.assertTrue(data["options"]["layout"]["hierarchical"]["enabled"])

    def test_repr_html_is_an_iframe(self):
        self.assertTrue(self.graph._repr_html_().startswith("<iframe"))

    def test_to_networkx(self):
        graph = self.graph.to_networkx()
        self.assertEqual(graph.number_of_nodes(), len(self.graph.nodes))
        self.assertEqual(graph.number_of_edges(), len(self.graph.edges))

    def test_decay_loops_can_be_drawn(self):
        net = load_dataset("FBNExampleNetworks")
        series = load_dataset("ExampleTimeseriesData")[0]
        graph = fbn_network_graph(net, type="dynamic", timeseries_matrix=series, to_time_point=3)
        self.assertIsNotNone(graph.draw())


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
        self.assertIsInstance(fbn_network_graph(self.network), FBNGraph)
        self.assertIsInstance(plot_network(self.network, type="static"), FBNGraph)

    def test_static_slice_no_error(self):
        graph = plot_network(
            self.network, type="staticSlice",
            timeseries_matrix=self.trainingseries[2], target_time_point=3,
        )
        self.assertIsInstance(graph, FBNGraph)
        self.assertIsInstance(fbn_network_graph(self.network, type="staticSlice", timeseries_matrix=self.trainingseries[2], to_time_point=3), FBNGraph)

    def test_dynamic_no_error(self):
        graph = plot_network(
            self.network, type="dynamic",
            timeseries_matrix=self.trainingseries[4], start_time_point=1, end_time_point=5,
        )
        self.assertIsInstance(graph, FBNGraph)
        self.assertIsNotNone(graph.draw())

    def test_draw_attractor_no_error(self):
        attractor = search_for_attractors(self.network, self.network["genes"], self.initial_states)
        graph = draw_attractor(self.network, attractor, index=1)
        self.assertIsInstance(graph, FBNGraph)
        self.assertTrue(graph.hierarchical)


if __name__ == "__main__":
    unittest.main()
