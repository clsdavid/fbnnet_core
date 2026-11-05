"""CEL -> genome-wide time series conversion, on tiny synthetic CDF/CEL files."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cel_fixtures import COLS, PROBE_SETS, write_cdf, write_cel
from fbnnet_core import bioinformatics, cel_import

class TestCelImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.cdf = root / "chip.cdf"
        write_cdf(cls.cdf)
        cls.cel_dir = root / "cel"
        cls.cel_dir.mkdir()
        rng = np.random.default_rng(1)
        # 3 patients: P1 and P2 have timepoints 0/6/24, P3 has 0/8/24; P4 is incomplete.
        for name in ("P1-0h", "P1-6h", "P1-24h", "P2-0h", "P2-6h", "P2-24h",
                     "P3-0h", "P3-8h", "P3-24h", "P4-0h", "P4-24h"):
            write_cel(cls.cel_dir / f"{name}.CEL", rng.uniform(50, 5000, COLS * COLS))
        # G1 is high at the last timepoint only (clear 2-means split); the other genes stay near
        # the same baseline with small noise, so quantile normalisation barely moves them.
        for patient, mid in (("P1", 6), ("P2", 6), ("P3", 8)):
            for hour in (0, mid, 24):
                values = rng.uniform(1000, 1100, COLS * COLS)
                values[[2, 4]] = 20000.0 if hour == 24 else rng.uniform(1000, 1100, 2)  # both G1 PM cells
                write_cel(cls.cel_dir / f"{patient}-{hour}h.CEL", values)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_cdf_pm_cells_only(self):
        names, cells = cel_import.read_cdf_pm_cells(self.cdf)
        self.assertEqual(names, list(PROBE_SETS))
        self.assertEqual([c.tolist() for c in cells], [[0], [2, 4], [6, 8], [10, 12]])

    def test_binary_cdf_is_rejected(self):
        bad = Path(self._tmp.name) / "bad.cdf"
        bad.write_bytes(b"\x43\x01\x00\x00binary")
        with self.assertRaises(ValueError):
            cel_import.read_cdf_pm_cells(bad)

    def test_cel_values_placed_by_coordinates(self):
        values = np.arange(COLS * COLS, dtype=float) * 10 + 1
        path = Path(self._tmp.name) / "ordered.CEL"
        write_cel(path, values)
        np.testing.assert_allclose(cel_import.read_cel_intensities(path), values)

    def test_binary_cel_is_rejected(self):
        bad = Path(self._tmp.name) / "bad.CEL"
        bad.write_bytes(b"\x40\x00\x00\x00binary")
        with self.assertRaises(ValueError):
            cel_import.read_cel_intensities(bad)

    def test_quantile_normalise_gives_identical_distributions(self):
        matrix = np.random.default_rng(2).normal(size=(50, 4)) * np.arange(1, 5)
        result = cel_import.quantile_normalise(matrix)
        reference = np.sort(result[:, 0])
        for column in range(1, 4):
            np.testing.assert_allclose(np.sort(result[:, column]), reference)
        np.testing.assert_array_equal(np.argsort(matrix[:, 1]), np.argsort(result[:, 1], kind="stable"))

    def test_quantile_normalise_averages_ties(self):
        matrix = np.array([[1.0, 5.0], [1.0, 1.0], [3.0, 3.0], [9.0, 9.0]])
        result = cel_import.quantile_normalise(matrix)
        reference = np.sort(matrix, axis=0).mean(axis=1)  # [1.0, 2.0, 4.0, 7.5]... per rank
        self.assertAlmostEqual(result[0, 0], result[1, 0])
        self.assertAlmostEqual(result[0, 0], (reference[0] + reference[1]) / 2)
        np.testing.assert_allclose(np.sort(result[:, 1]), reference)

    def test_grouping_orders_by_time_and_drops_incomplete_samples(self):
        groups = cel_import.group_cel_files(self.cel_dir)
        self.assertEqual(list(groups), ["P1", "P2", "P3"])
        self.assertEqual([p.name for p in groups["P3"]], ["P3-0h.CEL", "P3-8h.CEL", "P3-24h.CEL"])
        self.assertEqual([p.name for p in groups["P1"]], ["P1-0h.CEL", "P1-6h.CEL", "P1-24h.CEL"])

    def test_grouping_rejects_unmatched_directory(self):
        with self.assertRaises(ValueError):
            cel_import.group_cel_files(self.cel_dir, sample_pattern=r"^nomatch(?P<sample>x)(?P<time>1)$")

    def test_cel_to_timeseries_shape_and_controls(self):
        series = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        self.assertEqual(len(series), 3)
        for frame in series:
            self.assertEqual(list(frame.columns), ["t1", "t2", "t3"])
            self.assertEqual(list(frame.index), ["G1_at", "G2_at", "G3_at"])  # AFFX control dropped

    def test_binarised_output_is_boolean_and_top_variance_selection(self):
        series = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, n_genes=2)
        self.assertEqual(len(series[0].index), 2)
        values = np.stack([f.to_numpy() for f in series])
        self.assertTrue(set(np.unique(values)) <= {0.0, 1.0})
        self.assertTrue(all(f.loc["G1_at"].tolist() == [0.0, 0.0, 1.0] for f in series))

    def test_parallel_readers_match_single_reader_and_cache(self):
        cache = Path(self._tmp.name) / "expression.pkl"
        single = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        cached = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=2, binarise=False, cache_path=cache)
        again = cel_import.cel_to_timeseries(self.cel_dir, Path("missing.cdf"), binarise=False, cache_path=cache)
        for a, b, c in zip(single, cached, again):
            pd.testing.assert_frame_equal(a, b)
            pd.testing.assert_frame_equal(b, c)

    def test_save_and_load_roundtrip(self):
        series = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1)
        path = Path(self._tmp.name) / "series.npz"
        cel_import.save_timeseries(series, path)
        for original, loaded in zip(series, cel_import.load_timeseries(path)):
            pd.testing.assert_frame_equal(original, loaded)

    def test_median_polish_recovers_additive_model(self):
        rng = np.random.default_rng(3)
        probe_effect = rng.normal(size=(2, 5, 1))
        array_effect = rng.normal(size=(2, 1, 4)) + 8.0
        values = (probe_effect + array_effect).reshape(10, 4)
        summary = cel_import.median_polish_summary(values.copy(), np.array([5, 5]))
        centred = array_effect[:, 0, :] - np.median(array_effect[:, 0, :], axis=1, keepdims=True)
        np.testing.assert_allclose(summary - np.median(summary, axis=1, keepdims=True), centred, atol=1e-9)

    def test_median_polish_ignores_one_outlier_probe(self):
        clean = np.tile(np.array([5.0, 6.0, 7.0, 8.0]), (7, 1))
        corrupted = clean.copy()
        corrupted[0, 2] = 30.0
        summary = cel_import.median_polish_summary(corrupted, np.array([7]))
        np.testing.assert_allclose(summary[0], clean[0], atol=1e-9)

    def test_median_polish_groups_probe_sets_of_different_sizes(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(3 + 5 + 3, 4))
        counts = np.array([3, 5, 3])
        together = cel_import.median_polish_summary(values.copy(), counts)
        separate = np.vstack([
            cel_import.median_polish_summary(values[:3].copy(), np.array([3])),
            cel_import.median_polish_summary(values[3:8].copy(), np.array([5])),
            cel_import.median_polish_summary(values[8:].copy(), np.array([3])),
        ])
        np.testing.assert_allclose(together, separate)

    def test_mean_summary_option_and_validation(self):
        mean = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False, summary="mean")
        polish = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        self.assertEqual(mean[0].shape, polish[0].shape)
        with self.assertRaises(ValueError):
            cel_import.summarise_expression([], self.cdf, summary="nope")

    def test_pipeline_matches_stepwise_differential_selection_and_gene_mapping(self):
        # The toy chip has 7 PM cells, so absolute values are dominated by rank effects of the
        # quantile step; check that the options compose the documented steps instead.
        mapping = {"G1_at": "GENE_A", "G2_at": "GENE_A", "G3_at": "GENE_B"}
        options = dict(cutoff_induction=0.5, cutoff_repression=0.5, majority=2)
        raw = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        keep = bioinformatics.differentially_expressed_probesets(raw, **options)
        self.assertTrue(0 < len(keep) <= 3)
        expected = list(bioinformatics.convert_timeseries_probeset_name_to_gene_name(
            [f.loc[keep] for f in raw], mapping=mapping)["convert_data"].values())
        actual = cel_import.cel_to_timeseries(
            self.cel_dir, self.cdf, workers=1, binarise=False, map_genes=True, mapping=mapping, differential=options
        )
        for want, got in zip(expected, actual):
            pd.testing.assert_frame_equal(want, got)

    def test_differential_samples_option_restricts_the_tested_samples(self):
        options = dict(cutoff_induction=0.5, cutoff_repression=0.5, majority=2)
        raw = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        keep = bioinformatics.differentially_expressed_probesets(raw[:2], **options)
        actual = cel_import.cel_to_timeseries(
            self.cel_dir, self.cdf, workers=1, binarise=False,
            differential=dict(options, samples=["P1", "P2"]),
        )
        self.assertEqual(list(actual[0].index), keep)

    def test_cache_is_not_reused_for_another_summary(self):
        cache = Path(self._tmp.name) / "summary.pkl"
        mean = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False, summary="mean", cache_path=cache)
        polish = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False, cache_path=cache)
        direct = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False)
        pd.testing.assert_frame_equal(polish[0], direct[0])
        self.assertEqual(mean[0].shape, polish[0].shape)

    def test_result_feeds_cube_construction(self):
        from fbnnet_core.cube import construct_fbn_cube

        series = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1)
        genes = list(series[0].index)
        cube = construct_fbn_cube(genes, genes, series, max_k=2, temporal=1)
        self.assertIn("cube", cube)


if __name__ == "__main__":
    unittest.main()
