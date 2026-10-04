import unittest

import pandas as pd

from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.datasets import load_dataset
from fbnnet_core.network import merge_cluster_networks, mine_fbn_network_with_cores, search_fbn_core


class TestMergeClusterNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = load_dataset("ExampleTimeseriesData")
        cls.genes = list(cls.series[0].index)
        cube = construct_fbn_cube(cls.genes, cls.genes, cls.series, max_k=3, temporal=1, use_parallel=False)
        cls.cores_all = search_fbn_core(cube["cube"], cls.genes)
        cls.cores_first = {g: cls.cores_all[g] for g in cls.genes[:3] if g in cls.cores_all}
        cls.cores_last = {g: cls.cores_all[g] for g in cls.genes[2:] if g in cls.cores_all}

    def test_merging_two_clusters_equals_mining_all_cores(self):
        merged = merge_cluster_networks([
            {"NetworkCores": self.cores_first, "Genes": self.genes[:3]},
            {"NetworkCores": self.cores_last, "Genes": self.genes[2:]},
        ])
        expected = mine_fbn_network_with_cores(self.cores_all, self.genes)
        self.assertEqual(sorted(merged["genes"]), sorted(expected["genes"]))
        self.assertEqual(
            {g: sorted(r["expression"] for r in rules.values()) for g, rules in merged["interactions"].items()},
            {g: sorted(r["expression"] for r in rules.values()) for g, rules in expected["interactions"].items()},
        )

    def test_rules_found_in_several_clusters_are_kept_once(self):
        single = merge_cluster_networks([{"NetworkCores": self.cores_all, "Genes": self.genes}])
        twice = merge_cluster_networks([
            {"NetworkCores": self.cores_all, "Genes": self.genes},
            {"NetworkCores": self.cores_all, "Genes": self.genes},
        ])
        self.assertEqual(str(single), str(twice))

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            merge_cluster_networks([])
        with self.assertRaises(ValueError):
            merge_cluster_networks([{"Genes": []}])


if __name__ == "__main__":
    unittest.main()
