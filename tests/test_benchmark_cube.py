"""tools/benchmark_cube.py: statistics helpers, data generator and a tiny end-to-end run."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "benchmark_cube.py"
spec = importlib.util.spec_from_file_location("benchmark_cube", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules["benchmark_cube"] = bench
spec.loader.exec_module(bench)

from fbnnet_core import _parallel
from fbnnet_core import cube as cube_module  # noqa: E402

requires_parallel = unittest.skipUnless(_parallel.start_method(), "needs a multiprocessing start method")


def parse(*argv):
    parser = bench.build_parser()
    args = parser.parse_args(list(argv))
    bench.validate(args, parser)
    return args


class TestHelpers(unittest.TestCase):
    def test_tree_nodes_counts_both_branches_recursively(self):
        tree = {
            "A": {"SubGenesT": {"B": {}, "C": {"SubGenesF": {"D": {}}}}, "SubGenesF": {"E": {}}},
            "F": {},
        }
        self.assertEqual(bench.tree_nodes(tree), 6)
        self.assertEqual(bench.tree_nodes({}), 0)

    def test_survivor_stats_uses_root_sub_genes(self):
        cube = {"T1": {"SubGenes": {"A": {"SubGenesT": {"B": {}}}, "C": {}}}, "T2": {"SubGenes": {}}}
        stats = bench.survivor_stats(cube, 10)
        self.assertEqual(stats["T1"], {"S": 2, "S_over_G": 0.2, "nodes": 3})
        self.assertEqual(stats["T2"]["S"], 0)
        with_missing = bench.survivor_stats(cube, 10, ["T1", "T2", "T3"])
        self.assertEqual(with_missing["T3"], {"S": 0, "S_over_G": 0.0, "nodes": 0})

    def test_distribution(self):
        d = bench.distribution([4.0, 1.0, 3.0, 2.0, 10.0])
        self.assertEqual((d["min"], d["median"], d["max"]), (1.0, 3.0, 10.0))
        self.assertAlmostEqual(d["mean"], 4.0)
        self.assertEqual(bench.distribution([]), {})

    def test_fit_linear(self):
        fit = bench.fit_linear([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(fit["a"], 1.0)
        self.assertAlmostEqual(fit["b"], 2.0)
        self.assertAlmostEqual(fit["r2"], 1.0)
        self.assertIsNone(bench.fit_linear([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(bench.fit_linear([1, 2], [1, 2]))

    def test_parallel_lower_bound(self):
        self.assertEqual(bench.parallel_lower_bound([1, 1, 1, 1], 2), 2.0)  # average share dominates
        self.assertEqual(bench.parallel_lower_bound([9, 1, 1], 4), 9.0)    # the longest target dominates
        self.assertEqual(bench.parallel_lower_bound([], 4), 0.0)

    def test_human_duration(self):
        self.assertEqual(bench.human_duration(12), "12.0 s")
        self.assertEqual(bench.human_duration(600), "10.0 min")
        self.assertEqual(bench.human_duration(36000), "10.0 h")
        self.assertEqual(bench.human_duration(86400 * 5), "5.0 days")

    def test_compare_cubes_only_looks_at_common_targets(self):
        a, b = {"x": {"v": 1}, "y": {"v": 2}}, {"x": {"v": 1}}
        self.assertEqual(bench.compare_cubes(a, b), (True, 1))
        self.assertEqual(bench.compare_cubes(a, {"x": {"v": 9}}), (False, 1))


class TestSynthetic(unittest.TestCase):
    def test_shape_density_and_determinism(self):
        series = bench.make_synthetic(genes=50, samples=4, timepoints=12, density=0.3, seed=1)
        self.assertEqual(len(series), 4)
        self.assertEqual(series[0].shape, (50, 12))
        self.assertTrue(set(np.unique(series[0].to_numpy())) <= {0, 1})
        again = bench.make_synthetic(genes=50, samples=4, timepoints=12, density=0.3, seed=1)
        pd.testing.assert_frame_equal(series[2], again[2])
        mean = np.mean([s.to_numpy().mean() for s in series])
        self.assertAlmostEqual(mean, 0.3, delta=0.05)

    def test_planted_rule_is_present(self):
        planted = bench.make_synthetic(genes=30, samples=30, timepoints=30, planted=1, noise=0.0, seed=3)
        data = np.stack([s.to_numpy() for s in planted])  # samples x genes x time
        found = False
        for target in range(30):
            for a in range(30):
                for b in range(30):
                    if len({target, a, b}) == 3:
                        rule = data[:, a, :-1] & (1 - data[:, b, :-1])
                        if (data[:, target, 1:] == rule).all():
                            found = True
        self.assertTrue(found)


class TestEndToEnd(unittest.TestCase):
    def base(self, *extra):
        return parse("--data", "synthetic", "--genes", "120", "--samples", "6", "--timepoints", "12",
                     "--planted", "10", "--targets", "6", "--workers", "2", "--quiet", *extra)

    @requires_parallel
    def test_all_configurations_agree_and_report_statistics(self):
        report = bench.benchmark(self.base(), out=io.StringIO())
        names = [c["name"] for c in report["configs"]]
        self.assertEqual(names, ["tensor-seq", "tensor-par", "recursive-seq", "recursive-par"])
        self.assertTrue(all(c["identical_to_first"] for c in report["configs"]))
        self.assertTrue(all(c["compared_targets"] == len(report["targets"]) for c in report["configs"]))
        self.assertEqual(set(report["survivors"]), {"tensor", "recursive"})
        for backend in ("tensor", "recursive"):
            self.assertEqual(report["survivors"][backend]["S"], report["survivors"]["tensor"]["S"])
            seconds = report["per_target_time"][backend]["seconds"]
            self.assertGreater(seconds["max"], 0)
            self.assertEqual(len(report["per_target_time"][backend]["per_target"]), len(report["targets"]))
            self.assertGreater(report["cost_model"][backend]["extrapolated_seconds_with_workers"], 0)
            timing = report["per_target_time"][backend]
            self.assertGreaterEqual(timing["parallel_lower_bound_seconds"], timing["seconds"]["max"])
            self.assertEqual(timing["n_targets"], len(report["targets"]))
        par = next(c for c in report["configs"] if c["name"] == "tensor-par")
        self.assertEqual(par["workers"], 2)
        self.assertIsNotNone(par["efficiency"])

    def test_repeat_reports_the_median_of_several_runs(self):
        report = bench.benchmark(self.base("--configs", "tensor-seq", "--repeat", "3"), out=io.StringIO())
        self.assertEqual(len(report["configs"]), 1)
        self.assertEqual(len(report["per_target_time"]["tensor"]["per_target"]), len(report["targets"]))

    def test_recursive_is_skipped_above_the_gene_limit_and_can_use_a_subset(self):
        skipped = bench.benchmark(self.base("--configs", "tensor-seq", "recursive-seq", "--max-recursive-genes", "50"),
                                  out=io.StringIO())
        self.assertEqual(skipped["skipped"], ["recursive-seq"])
        self.assertEqual([c["name"] for c in skipped["configs"]], ["tensor-seq"])
        subset = bench.benchmark(self.base("--configs", "tensor-seq", "recursive-seq", "--recursive-targets", "2"),
                                 out=io.StringIO())
        recursive = next(c for c in subset["configs"] if c["name"] == "recursive-seq")
        self.assertEqual(recursive["n_targets"], 2)
        self.assertEqual(recursive["compared_targets"], 2)
        self.assertTrue(recursive["identical_to_first"])

    def test_cli_writes_json_and_csv_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path, csv_path = Path(folder) / "r.json", Path(folder) / "r.csv"
            code = bench.main(["--data", "synthetic", "--genes", "80", "--samples", "5", "--timepoints", "10",
                               "--targets", "4", "--configs", "tensor-seq", "recursive-seq", "--quiet",
                               "--json", str(json_path), "--csv", str(csv_path)])
            self.assertEqual(code, 0)
            report = json.loads(json_path.read_text())
            self.assertEqual(report["data"]["genes"], 80)
            table = pd.read_csv(csv_path)
            self.assertEqual(set(table.columns), {"backend", "target", "S", "S_over_G", "nodes", "seconds"})
            self.assertEqual(len(table), 2 * len(report["targets"]))  # every kept target, both engines, zero-survivor ones included

    def test_argument_validation(self):
        for argv in (["--data", "npz"], ["--data", "cel", "--cel-dir", "x"], ["--targets", "0"]):
            with self.assertRaises(SystemExit):
                parse(*argv)

    def test_leukaemia_data_source(self):
        args = parse("--data", "leukaemia", "--targets", "3", "--configs", "tensor-seq", "--quiet")
        report = bench.benchmark(args, out=io.StringIO())
        self.assertEqual(report["data"]["genes"], 285)
        self.assertEqual(len(report["survivors"]["tensor"]["per_target"]), 3)


if __name__ == "__main__":
    unittest.main()
