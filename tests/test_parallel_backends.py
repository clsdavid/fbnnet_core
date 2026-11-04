"""Parallel execution and mining backends must not change results."""
import unittest
import warnings

import pandas as pd

from fbnnet_core import cube as cube_module
from fbnnet_core import generate_fbm_network, load_network, mine_fbn_network
from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries

requires_fork = unittest.skipUnless(cube_module._FORK_AVAILABLE, "needs the fork start method")


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

    @requires_fork
    def test_parallel_matches_sequential_for_both_backends(self):
        for backend in ("recursive", "tensor"):
            with self.subTest(backend=backend):
                sequential = self.build(backend=backend)
                parallel = self.build(backend=backend, use_parallel=True, max_workers=2, chunksize=1)
                self.assertEqual(sequential, parallel)

    def test_single_worker_runs_sequentially(self):
        self.assertEqual(self.build(), self.build(use_parallel=True, max_workers=1))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(backend="gpu")
        with self.assertRaises(ValueError):
            cube_module.do_non_parallel_work(self.genes[:1], self.genes, 2, 1, {}, backend="gpu")

    def test_mine_fbn_network_warns_that_use_parallel_is_ignored(self):
        cube = self.build()
        self.assertTrue(cube["cube"])
        expected = mine_fbn_network(cube, use_parallel=False)
        with self.assertWarnsRegex(UserWarning, "use_parallel is not supported"):
            actual = mine_fbn_network(cube, use_parallel=True)
        self.assertEqual(expected, actual)

    @requires_fork
    def test_generate_fbm_network_parallel_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            parallel = generate_fbm_network(
                self.samples, max_k=3, use_parallel=True, max_workers=2, backend="tensor"
            )
        sequential = generate_fbm_network(self.samples, max_k=3)
        self.assertTrue(sequential["interactions"])
        self.assertEqual(sequential, parallel)


if __name__ == "__main__":
    unittest.main()
