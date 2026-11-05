"""Parallel execution and mining backends must not change results."""
import os
import time
import unittest
import warnings
from unittest import mock

import pandas as pd

from fbnnet_core import _parallel
from fbnnet_core import cube as cube_module
from fbnnet_core.attractor import reconstruct_timeseries
from fbnnet_core import generate_fbm_network, load_network, mine_fbn_network
from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries

requires_parallel = unittest.skipUnless(_parallel.start_method(), "needs a multiprocessing start method")


class TestParallelAndBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Time series with real rules, so the cubes and networks compared below are not empty.
        definition = (
            "targets, factors\n"
            "Gene1, Gene1\n"
            "Gene2, Gene1 & Gene5 & !Gene4\n"
            "Gene3, Gene3\n"
            "Gene4, Gene3 & !(Gene1 & Gene5)\n"
            "Gene5, !Gene2\n"
        )
        with open("example.bn", "w") as handle:
            handle.write(definition)
        network = load_network("example.bn")
        cls.genes = list(network["genes"])
        series = generateBoolNetTimeseries(
            network, generateAllCombinationBinary(cls.genes), 12, transition_type="synchronous"
        )
        cls.samples = [
            pd.DataFrame(mat, index=cls.genes, columns=[str(j + 1) for j in range(mat.shape[1])])
            for mat in series
        ]

    def build(self, **kwargs):
        kwargs.setdefault("max_k", 3)
        kwargs.setdefault("temporal", 1)
        return construct_fbn_cube(self.genes, self.genes, self.samples, **kwargs)

    def test_tensor_backend_matches_recursive(self):
        for temporal in (1, 2):
            with self.subTest(temporal=temporal):
                self.assertEqual(
                    self.build(temporal=temporal),
                    self.build(temporal=temporal, backend="tensor"),
                )

    @requires_parallel
    def test_parallel_matches_sequential_for_both_backends(self):
        for backend in ("recursive", "tensor"):
            with self.subTest(backend=backend):
                sequential = self.build(backend=backend)
                parallel = self.build(backend=backend, use_parallel=True, max_workers=2, chunksize=1)
                self.assertEqual(sequential, parallel)

    @requires_parallel
    def test_cellcycle_wall_clock(self):
        # Informational benchmark: prints timings, asserts only result parity.
        path = os.path.join(os.path.dirname(__file__), "..", "example", "cellcycle.bn")
        network = load_network(path)
        genes = list(network["genes"])
        series = generateBoolNetTimeseries(
            network, generateAllCombinationBinary(genes)[:64], 12, transition_type="synchronous"
        )
        samples = [
            pd.DataFrame(mat, index=genes, columns=[str(j + 1) for j in range(mat.shape[1])])
            for mat in series
        ]
        timings = {}
        cubes = {}
        for label, kwargs in (("sequential", {}), ("parallel", {"use_parallel": True})):
            start = time.perf_counter()
            cubes[label] = construct_fbn_cube(genes, genes, samples, max_k=2, temporal=1, **kwargs)
            timings[label] = time.perf_counter() - start
        print(
            f"cellcycle ({len(genes)} genes, {cube_module.DEFAULT_MAX_WORKERS} workers): "
            f"sequential {timings['sequential']:.2f}s, parallel {timings['parallel']:.2f}s, "
            f"speedup {timings['sequential'] / timings['parallel']:.2f}x"
        )
        self.assertEqual(cubes["sequential"], cubes["parallel"])

    def test_single_worker_runs_sequentially(self):
        self.assertEqual(self.build(), self.build(use_parallel=True, max_workers=1))

    def test_auto_backend_resolves_to_tensor_and_is_the_default(self):
        self.assertEqual(cube_module.resolve_backend("auto"), "tensor")
        self.assertEqual(cube_module.resolve_backend("recursive"), "recursive")
        reference = self.build(backend="recursive")
        self.assertEqual(self.build(backend="auto"), reference)
        self.assertEqual(self.build(backend="auto", use_parallel=True, max_workers=2, chunksize=1), reference)
        self.assertEqual(self.build(), reference)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(backend="gpu")
        with self.assertRaises(ValueError):
            cube_module.do_non_parallel_work(self.genes[:1], self.genes, 2, 1, {}, backend="gpu")

    @requires_parallel
    def test_parallel_rule_mining_matches_sequential(self):
        cube = self.build()
        self.assertTrue(cube["cube"])
        expected = mine_fbn_network(cube)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self.assertEqual(expected, mine_fbn_network(cube, use_parallel=True, max_workers=2))
            self.assertEqual(expected, mine_fbn_network(cube, use_parallel=True, max_workers=3, chunksize=1))
        # one worker (or no fork) runs in-process
        self.assertEqual(expected, mine_fbn_network(cube, use_parallel=True, max_workers=1))

    @requires_parallel
    def test_generate_fbm_network_parallel_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            parallel = generate_fbm_network(
                self.samples, max_k=3, use_parallel=True, max_workers=2, backend="tensor"
            )
        sequential = generate_fbm_network(self.samples, max_k=3)
        self.assertTrue(sequential["interactions"])
        self.assertEqual(sequential, parallel)


class TestSpawnStartMethod(TestParallelAndBackends):
    """Everything above again with the spawn start method (the only one on Windows)."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {_parallel.START_METHOD_ENV: "spawn"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertEqual(_parallel.start_method(), "spawn")

    def test_reconstruct_timeseries_matches_sequential(self):
        network = generate_fbm_network(self.samples, max_k=3)
        states = [dict(zip(self.genes, bits)) for bits in ([0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1])]
        sequential = reconstruct_timeseries(network, states, max_timepoints=8)
        parallel = reconstruct_timeseries(network, states, max_timepoints=8, use_parallel=True, max_workers=2)
        for expected, actual in zip(sequential, parallel):
            self.assertTrue((expected == actual).all())

    def test_spawn_without_time_series_falls_back_to_sequential(self):
        main_parameters = cube_module.convert_df_main_parameters(self.samples, 1)
        with self.assertLogs("fbnnet_core.cube", level="WARNING"):
            result = cube_module.do_parallel_work(
                self.genes, self.genes, 2, 1, main_parameters, max_workers=2, backend="tensor"
            )
        expected = cube_module.do_non_parallel_work(self.genes, self.genes, 2, 1, main_parameters, "tensor")
        self.assertEqual(result, expected)


class TestStartMethodSelection(unittest.TestCase):
    def test_override_and_fallback(self):
        with mock.patch.dict(os.environ, {_parallel.START_METHOD_ENV: "spawn"}):
            self.assertEqual(_parallel.start_method(), "spawn")
        with mock.patch.dict(os.environ, {_parallel.START_METHOD_ENV: ""}):
            default = _parallel.start_method()
        with mock.patch.dict(os.environ, {_parallel.START_METHOD_ENV: "bogus"}):
            self.assertEqual(_parallel.start_method(), default)
        with mock.patch.object(_parallel.multiprocessing, "get_all_start_methods", return_value=["spawn"]):
            self.assertEqual(_parallel.start_method(), "spawn")
        with mock.patch.object(_parallel.multiprocessing, "get_all_start_methods", return_value=[]):
            self.assertIsNone(_parallel.start_method())


if __name__ == "__main__":
    unittest.main()
