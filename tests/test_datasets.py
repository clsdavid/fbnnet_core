import unittest

import pandas as pd

from fbnnet_core import available_datasets, load_dataset
from fbnnet_core.attractor import search_for_attractors
from fbnnet_core.fbn_types import FundamentalBooleanNetwork


class TestDatasets(unittest.TestCase):
    def test_every_dataset_loads(self):
        for name in available_datasets():
            with self.subTest(name=name):
                self.assertIsNotNone(load_dataset(name))

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            load_dataset("nope")

    def test_leukaemia_networks(self):
        for name, genes in (("FBM_Leukeamia_Networks", 285), ("TFBM_Leukeamia_Networks", 285), ("Leukeamia_Networks", 285)):
            network = load_dataset(name)
            self.assertIsInstance(network, FundamentalBooleanNetwork)
            self.assertEqual(len(network["genes"]), genes)
            self.assertEqual(network["class"], "FundamentalBooleanNetwork")
        fbm = load_dataset("FBM_Leukeamia_Networks")
        timesteps = {rule["timestep"] for rules in fbm["interactions"].values() for rule in rules.values()}
        self.assertEqual(timesteps, {1.0})
        tfbm = load_dataset("TFBM_Leukeamia_Networks")
        timesteps = {rule["timestep"] for rules in tfbm["interactions"].values() for rule in rules.values()}
        self.assertEqual(timesteps, {1.0, 2.0})
        self.assertEqual(len(load_dataset("fbm_leukeamia_network")["genes"]), 285)

    def test_rules_have_numeric_fields_and_list_inputs(self):
        network = load_dataset("FBNExampleNetworks")
        rule = network["interactions"]["Gene2"]["Gene2_1_Activator"]
        self.assertEqual(rule["input"], [1, 4, 5])
        self.assertEqual(rule["expression"], "Gene1&!Gene4&Gene5")
        self.assertEqual((rule["type"], rule["probability"], rule["timestep"]), (1.0, 1.0, 1.0))
        self.assertIn("Fundamental Boolean Network with  5 genes", str(network))

    def test_example_network_attractors(self):
        network = load_dataset("FBNExampleNetworks")
        attractors = search_for_attractors(network, network["genes"])
        self.assertGreater(len(attractors["Attractors"]), 0)

    def test_time_series(self):
        series = load_dataset("Leukeamia_Timeseries")
        self.assertEqual(len(series), 26)
        self.assertEqual(series[0].shape, (285, 3))
        self.assertEqual(series[0].attrs["sample"], "B-ALL-13")
        example = load_dataset("ExampleTimeseriesData")
        self.assertEqual(len(example), 32)
        self.assertEqual(example[0].shape, (5, 43))
        self.assertIsInstance(load_dataset("yeastTimeSeries"), pd.DataFrame)

    def test_boolnet_networks(self):
        network = load_dataset("ExampleNetwork")
        self.assertEqual(network["genes"], ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"])
        self.assertEqual(network["expressions"]["Gene5"], "!Gene2")

    def test_annotation_and_gene_lists(self):
        self.assertEqual(len(load_dataset("Common_Genes_Leukeamia")), 285)
        david = load_dataset("DAVID_Gene_List")
        self.assertEqual(list(david.columns), ["Probeset", "ENTREZ_Gene_ID", "Species", "Gene.Name", "Symbol"])
        clusters = load_dataset("Common_Genes_Clusters")
        self.assertEqual(len(clusters), 44)


if __name__ == "__main__":
    unittest.main()
