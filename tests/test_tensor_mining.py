# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Leshi Chen
import math
import unittest

import pandas as pd
from fbnnet_core import _core
from fbnnet_core import _tree

from fbnnet_core.boolnet import load_network
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from fbnnet_core.cube import convert_df_main_parameters
from fbnnet_core.tensor_mining import batched_basic_measures, gene_probabilities_measurements


class TestTensorMining(unittest.TestCase):
    """Validates the Phase 3 node-level candidate-gene batching backend
    (fbnnet_core/tensor_mining.py) against the ground-truth C++ engine
    (_core.getGeneProbabilities_basic, _tree.getGeneProbabilities_measurements).
    """

    @classmethod
    def setUpClass(cls):
        with open("example.bn", "w") as f:
            f.write("targets, factors\n")
            f.write("Gene1, Gene1\n")
            f.write("Gene2, Gene1 & Gene5 & !Gene4\n")
            f.write("Gene3, Gene3\n")
            f.write("Gene4, Gene3 & !(Gene1 & Gene5)\n")
            f.write("Gene5, !Gene2\n")

        network = load_network("example.bn")
        initial_states = generateAllCombinationBinary(network["genes"])
        trainingseries = generateBoolNetTimeseries(network, initial_states, 43, transition_type="synchronous")
        cls.genes = list(network["genes"])
        cls.trainingseries_df = [
            pd.DataFrame(mat, index=cls.genes, columns=[str(j + 1) for j in range(mat.shape[1])])
            for mat in trainingseries
        ]

    def test_batched_basic_measures_matches_cpp(self):
        for temporal in (1, 2, 3):
            main_parameters = convert_df_main_parameters(self.trainingseries_df, temporal=temporal)
            for target_gene in self.genes:
                candidates = [g for g in self.genes if g != target_gene]

                with self.subTest(temporal=temporal, target=target_gene, case="no_fixed"):
                    fixed_state = {}
                    batched = batched_basic_measures(main_parameters, target_gene, candidates, fixed_state, temporal)
                    for cand in candidates:
                        ref = _core.getGeneProbabilities_basic(
                            main_parameters, None, [target_gene], [cand], temporal
                        )
                        for ts_key, ref_dict in ref.items():
                            got = batched[cand][ts_key]
                            for k, v in ref_dict.items():
                                self.assertEqual(
                                    got[k], v,
                                    f"temporal={temporal} target={target_gene} cand={cand} ts={ts_key} key={k}",
                                )

                if len(candidates) >= 2:
                    with self.subTest(temporal=temporal, target=target_gene, case="one_fixed"):
                        fixed_gene = candidates[0]
                        fixed_state2 = {fixed_gene: 1}
                        remaining = [g for g in candidates if g != fixed_gene]
                        batched2 = batched_basic_measures(
                            main_parameters, target_gene, remaining, fixed_state2, temporal
                        )
                        for cand in remaining:
                            ref = _core.getGeneProbabilities_basic(
                                main_parameters, dict(fixed_state2), [target_gene], [cand], temporal
                            )
                            for ts_key, ref_dict in ref.items():
                                got = batched2[cand][ts_key]
                                for k, v in ref_dict.items():
                                    self.assertEqual(
                                        got[k], v,
                                        f"temporal={temporal} target={target_gene} cand={cand} ts={ts_key} key={k}",
                                    )

    def test_gene_probabilities_measurements_matches_cpp(self):
        for temporal in (1, 2):
            main_parameters = convert_df_main_parameters(self.trainingseries_df, temporal=temporal)
            for target_gene in self.genes:
                candidates = [g for g in self.genes if g != target_gene]
                with self.subTest(temporal=temporal, target=target_gene):
                    ref = _tree.getGeneProbabilities_measurements(
                        [target_gene], main_parameters, candidates, None, temporal, False
                    )
                    got = gene_probabilities_measurements(
                        main_parameters, target_gene, candidates, {}, temporal, False
                    )
                    self.assertEqual(set(ref.keys()), set(got.keys()))
                    for gene in ref.keys():
                        ref_p = ref[gene]["probabilityOfFourCombines_P"]
                        ref_n = ref[gene]["probabilityOfFourCombines_N"]
                        got_p = got[gene]["probabilityOfFourCombines_P"]
                        got_n = got[gene]["probabilityOfFourCombines_N"]
                        for k in ref_p.keys():
                            self.assertEqual(ref_p[k], got_p[k], f"P key={k} gene={gene} temporal={temporal}")
                        for k in ref_n.keys():
                            self.assertEqual(ref_n[k], got_n[k], f"N key={k} gene={gene} temporal={temporal}")


class TestTensorNodesAgainstReference(unittest.TestCase):
    """The columnar node evaluation must equal the per-candidate reference on noisy random data."""

    def test_matches_reference_with_fixed_genes_and_basic_measures(self):
        import numpy as np
        from fbnnet_core import tensor_mining

        rng = np.random.default_rng(7)
        genes = [f"G{i}" for i in range(40)]
        data = (rng.random((6, 40, 12)) < 0.45).astype(int)
        data[:, 1, 1:] = data[:, 2, :-1] & (1 - data[:, 3, :-1])
        series = [
            pd.DataFrame(m, index=genes, columns=[str(j + 1) for j in range(m.shape[1])]) for m in data
        ]
        for temporal in (1, 2, 3):
            main_parameters = convert_df_main_parameters(series, temporal=temporal)
            for fixed_state in ({}, {"G2": 1}, {"G2": 1, "G3": 0}):
                for show_basic in (False, True):
                    for target in ("G1", "G7"):
                        candidates = [g for g in genes if g != target and g not in fixed_state]
                        with self.subTest(temporal=temporal, fixed=fixed_state, basic=show_basic, target=target):
                            ref = tensor_mining.gene_probabilities_measurements_reference(
                                main_parameters, target, candidates, fixed_state, temporal, show_basic
                            )
                            got = gene_probabilities_measurements(
                                main_parameters, target, candidates, fixed_state, temporal, show_basic
                            )
                            self.assertEqual(list(ref.keys()), list(got.keys()))
                            self.assertEqual(ref, got)

    def test_empty_candidates(self):
        main_parameters = convert_df_main_parameters(
            [pd.DataFrame([[0, 1, 0], [1, 0, 1]], index=["A", "B"], columns=["1", "2", "3"])], temporal=1
        )
        self.assertEqual(gene_probabilities_measurements(main_parameters, "A", [], {}, 1), {})


def _same_value(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (a != a and b != b) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    return type(a) is type(b) and a == b


class TestEssentialCandidateFilter(unittest.TestCase):
    """`has_essential_candidate` must keep exactly the targets `filterTargetGenesByConditionGenes` keeps."""

    def check(self, series, temporals):
        from fbnnet_core import tensor_nodes

        genes = list(series[0].index)
        for temporal in temporals:
            main_parameters = convert_df_main_parameters(series, temporal)
            expected = set(_tree.filterTargetGenesByConditionGenes(genes, main_parameters, genes, None, temporal))
            cache = tensor_nodes.prepare(main_parameters, temporal)
            actual = {g for g in genes if tensor_nodes.has_essential_candidate(cache, g, genes, temporal)}
            with self.subTest(temporal=temporal, genes=len(genes)):
                self.assertEqual(expected, actual)
        return expected

    def test_random_data_with_planted_rules(self):
        import numpy as np

        rng = np.random.default_rng(5)
        data = (rng.random((8, 50, 10)) < 0.45).astype(int)
        for target, a, b in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
            data[:, target, 1:] = data[:, a, :-1] & (1 - data[:, b, :-1])
        series = [
            pd.DataFrame(m, index=[f"G{i}" for i in range(50)], columns=[str(j + 1) for j in range(10)])
            for m in data
        ]
        kept = self.check(series, (1, 2, 3))
        self.assertTrue({"G1", "G4", "G7"} <= kept)

    def test_small_sample_with_rejected_targets(self):
        import numpy as np

        rng = np.random.default_rng(9)
        data = (rng.random((4, 60, 3)) < 0.4).astype(int)  # too little data for most genes to be essential
        series = [
            pd.DataFrame(m, index=[f"G{i}" for i in range(60)], columns=["1", "2", "3"]) for m in data
        ]
        kept = self.check(series, (1,))
        self.assertTrue(0 < len(kept) < 60)

    def test_no_genes(self):
        from fbnnet_core import tensor_nodes

        main_parameters = convert_df_main_parameters(
            [pd.DataFrame([[0, 1, 0], [1, 0, 1]], index=["A", "B"], columns=["1", "2", "3"])], temporal=1
        )
        cache = tensor_nodes.prepare(main_parameters, 1)
        self.assertFalse(tensor_nodes.has_essential_candidate(cache, "A", [], 1))


class TestAdvancedMeasuresBatch(unittest.TestCase):
    """`getAdvancedMeasuresBatch` and `getAdvancedMeasures` share one C++ implementation: identical bits."""

    @staticmethod
    def basic(row, target_t, target_f, total, step):
        tt, ft, tf, ff, tt_c, ft_c, tf_c, ff_c = (int(v) for v in row)
        return {
            "target_T_count": target_t, "target_F_count": target_f,
            "cond_T_count": tt + tf, "cond_F_count": ft + ff,
            "cond_T_count_c": tt_c + tf_c, "cond_F_count_c": ft_c + ff_c,
            "count_cond_T_target_T": tt, "count_cond_F_target_T": ft,
            "count_cond_T_target_F": tf, "count_cond_F_target_F": ff,
            "count_cond_T_target_T_c": tt_c, "count_cond_F_target_T_c": ft_c,
            "count_cond_T_target_F_c": tf_c, "count_cond_F_target_F_c": ff_c,
            "total_calculated_timepoints": total, "num_of_conditional_genes": 1, "timestep": step,
        }

    def test_batch_equals_dict_function_on_random_and_edge_rows(self):
        import numpy as np
        from fbnnet_core import tensor_nodes

        rng = np.random.default_rng(11)
        for total, target_t, target_f, step in ((26, 0, 0, 1), (26, 5, 9, 1), (39, 13, 20, 2), (300, 140, 90, 1)):
            high = max(2, total // 3)
            rows = rng.integers(0, high, size=(400, 8))
            rows[:20] = 0
            rows[20:40, :4] = 0
            rows[40:60, 4:] = 0
            rows[60:80] = rng.integers(0, 2, size=(20, 8))
            batch = _core.getAdvancedMeasuresBatch(rows.astype(np.int64), target_t, target_f, total, step)
            lazy = tensor_nodes._LazyMeasures(batch, rows, {"timestep": step}, False)
            for i, row in enumerate(rows):
                expected = _core.getAdvancedMeasures(self.basic(row, target_t, target_f, total, step), False)
                built = lazy[i]
                with self.subTest(total=total, row=row.tolist()):
                    self.assertEqual(list(built), list(expected))
                    for key, value in expected.items():
                        self.assertTrue(_same_value(built[key], value), f"{key}: {built[key]!r} != {value!r}")

    def test_show_basic_measures_is_reproduced(self):
        import numpy as np
        from fbnnet_core import tensor_nodes

        rows = np.array([[3, 2, 1, 4, 2, 2, 1, 5]], dtype=np.int64)
        constants = {"target_T_count": 4, "target_F_count": 6, "total_calculated_timepoints": 26,
                     "num_of_conditional_genes": 1, "timestep": 1}
        batch = _core.getAdvancedMeasuresBatch(rows, 4, 6, 26, 1)
        built = tensor_nodes._LazyMeasures(batch, rows, constants, True)[0]
        expected = _core.getAdvancedMeasures(self.basic(rows[0], 4, 6, 26, 1), True)
        self.assertEqual(list(built), list(expected))
        self.assertEqual(built["basic_measures"], expected["basic_measures"])

    def test_batch_rejects_wrong_shape_and_handles_no_rows(self):
        import numpy as np

        with self.assertRaises(ValueError):
            _core.getAdvancedMeasuresBatch(np.zeros((3, 7), dtype=np.int64), 1, 1, 10, 1)
        empty = _core.getAdvancedMeasuresBatch(np.zeros((0, 8), dtype=np.int64), 1, 1, 10, 1)
        self.assertTrue(all(len(column) == 0 for column in empty.values()))


if __name__ == "__main__":
    unittest.main()
