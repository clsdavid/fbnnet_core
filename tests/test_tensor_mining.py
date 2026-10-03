import unittest

import pandas as pd
import fbnnet_core
import fbnnet_tree

from py_src.boolnet import load_network
from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from py_src.cube import convert_df_main_parameters
from py_src.tensor_mining import batched_basic_measures, gene_probabilities_measurements


class TestTensorMining(unittest.TestCase):
    """Validates the Phase 3 node-level candidate-gene batching backend
    (py_src/tensor_mining.py) against the ground-truth C++ engine
    (fbnnet_core.getGeneProbabilities_basic, fbnnet_tree.getGeneProbabilities_measurements).
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
                        ref = fbnnet_core.getGeneProbabilities_basic(
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
                            ref = fbnnet_core.getGeneProbabilities_basic(
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
                    ref = fbnnet_tree.getGeneProbabilities_measurements(
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


if __name__ == "__main__":
    unittest.main()
