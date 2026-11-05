"""Ports of R/FBNBioinformatics.R and R/sample_FBN.R (see fbnnet_core.bioinformatics)."""
import contextlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from cel_fixtures import write_cdf, write_cel
from fbnnet_core import bioinformatics as bio
from fbnnet_core import cel_import

ANNOTATION = pd.DataFrame({
    "PROBEID": ["p1", "p2", "p3", "p4", "p5", "p6", "p1"],
    "SYMBOL": ["AAA", "AAA", "BBB", None, "CCC", "DDD", "AAA_dup"],
    "ENTREZID": ["20", "20", "10", "30", "40", None, "20"],
    "GENENAME": ["gene a", "gene a", "gene b", "gene d", "gene c", "gene d", "gene a"],
})


def samples(values, index=("g1", "g2", "g3"), columns=("t1", "t2", "t3")):
    return {name: pd.DataFrame(v, index=list(index), columns=list(columns)) for name, v in values.items()}


class TestSampleGrouping(unittest.TestCase):
    def setUp(self):
        columns = ["B-ALL-13-24h.CEL", "B-ALL-13-0h.CEL", "B-ALL-13-8h.CEL", "T-ALL-2-0h.CEL", "T-ALL-2-24h.CEL"]
        self.matrix = pd.DataFrame(np.arange(10.0).reshape(2, 5), index=["z_at", "a_at"], columns=columns)

    def test_convert_into_sample_timeseries_groups_by_first_three_pieces(self):
        groups = bio.convert_into_sample_timeseries(self.matrix)
        self.assertEqual(list(groups), ["B-ALL-13", "T-ALL-2"])
        self.assertEqual(groups["B-ALL-13"].shape, (2, 3))
        custom = bio.convert_into_sample_timeseries(self.matrix, func=lambda p: p[0], splitor="-")
        self.assertEqual(list(custom), ["B", "T"])

    def test_reorder_orders_columns_by_hour_and_rows_by_name(self):
        ordered = bio.reorder_sample_timeseries(bio.convert_into_sample_timeseries(self.matrix))
        frame = ordered["B-ALL-13"]
        self.assertEqual(list(frame.columns), [0, 8, 24])
        self.assertEqual(list(frame.index), ["a_at", "z_at"])
        self.assertEqual(frame.loc["z_at"].tolist(), [1.0, 2.0, 0.0])
        self.assertEqual(list(ordered["T-ALL-2"].columns), [0, 24])

    def test_reorder_accepts_a_list_and_rejects_names_without_a_time(self):
        frame = pd.DataFrame([[1, 2]], index=["g"], columns=["S-8h", "S-0h"])
        self.assertEqual(list(bio.reorder_sample_timeseries([frame])["sample1"].columns), [0, 8])
        with self.assertRaises(ValueError):
            bio.reorder_sample_timeseries({"x": pd.DataFrame([[1]], columns=["S-none"])})

    def test_dissolve_flattens_nested_containers(self):
        nested = [{"a": 1, "b": {"c": 2}}, {"d": 3}, [4, 5]]
        self.assertEqual(bio.dissolve(nested), {"a": 1, "c": 2, "d": 3, "3.1": 4, "3.2": 5})


class TestAnnotation(unittest.TestCase):
    def test_mapping_object_follows_r_deduplication(self):
        mapped = bio.map_to_gene_name_with_hgu133plus2_db(["p1", "p2", "p3", "p4", "p5", "p6"], ANNOTATION)
        # p4 (no symbol... NaN) and p6 (no Entrez id) are dropped; p1 is deduplicated by probe set
        self.assertEqual(sorted(mapped.MappedProbeset["PROBEID"]), ["p1", "p2", "p3", "p5"])
        self.assertEqual(mapped.MappedProbeset.set_index("PROBEID").loc["p1", "SYMBOL"], "AAA")
        # ordered by Entrez id; one representative probe set per Entrez id
        self.assertEqual(mapped.DeDuplicatedMappedProbeset["ENTREZID"].tolist(), ["10", "20", "40"])
        sheet = dict(zip(mapped.DeDuplicatedMapping["original"], mapped.DeDuplicatedMapping["mapTo"]))
        self.assertEqual(sheet["p1"], sheet["p2"])
        self.assertEqual(sheet["p3"], "p3")

    def test_nothing_mapped_gives_none_and_missing_columns_are_rejected(self):
        self.assertIsNone(bio.map_to_gene_name_with_hgu133plus2_db(["nope"], ANNOTATION))
        with self.assertRaises(ValueError):
            bio.map_to_gene_name_with_hgu133plus2_db(["p1"], ANNOTATION[["PROBEID", "SYMBOL"]])

    def test_default_annotation_is_the_bundled_david_list(self):
        from fbnnet_core.datasets import load_dataset

        david = load_dataset("DAVID_Gene_List")
        probeset, symbol = david.iloc[0]["Probeset"], david.iloc[0]["Symbol"]
        mapped = bio.map_to_gene_name_with_hgu133plus2_db([probeset])
        self.assertEqual(mapped.MappedProbeset.iloc[0]["SYMBOL"], symbol)

    def test_convert_probeset_to_gene_returns_the_representative_row(self):
        mapped = bio.map_to_gene_name_with_hgu133plus2_db(["p1", "p2", "p3"], ANNOTATION)
        rows = bio.convert_probeset_to_gene(mapped, "p2")
        self.assertEqual(rows["SYMBOL"].tolist(), ["AAA"])
        self.assertTrue(bio.convert_probeset_to_gene(mapped, "unknown").empty)
        with self.assertRaises(TypeError):
            bio.convert_probeset_to_gene({"x": 1}, "p1")


class TestMapProbesetNames(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(bio.map_probeset_names(None), {})
        self.assertEqual(bio.map_probeset_names([]), {})

    def test_primary_mapping_wins_then_david_then_na(self):
        from fbnnet_core.datasets import load_dataset

        david = load_dataset("DAVID_Gene_List")
        known, symbol = david.iloc[0]["Probeset"], david.iloc[0]["Symbol"]
        result = bio.map_probeset_names([known, "p_primary", "no_such"], {"p_primary": "GENE_P"})
        self.assertEqual(result, {known: [symbol], "p_primary": ["GENE_P"], "no_such": ["N/A"]})
        self.assertEqual(bio.map_probeset_names([known], {known: "FIRST"}), {known: ["FIRST"]})

    def test_dataframe_and_annotation_object_mappings(self):
        frame = pd.DataFrame({"Probeset": ["p1", "p1", "p2"], "Symbol": ["A", "B", None]})
        result = bio.map_probeset_names(["p1", "p2"], frame)
        self.assertEqual(result, {"p1": ["A", "B"], "p2": ["N/A"]})
        mapped = bio.map_to_gene_name_with_hgu133plus2_db(["p1", "p2", "p3"], ANNOTATION)
        self.assertEqual(bio.map_probeset_names(["p2", "p3"], mapped), {"p2": ["AAA"], "p3": ["BBB"]})


class TestDifferentialExpression(unittest.TestCase):
    UP, DOWN, FLAT = [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]

    def series(self):
        index = ["up", "down", "flat", "mixed"]
        rows = [
            [self.UP, self.DOWN, self.FLAT, self.UP],
            [self.UP, self.DOWN, self.FLAT, self.UP],
            [self.UP, self.DOWN, self.FLAT, self.DOWN],
            [self.UP, self.DOWN, self.FLAT, self.DOWN],
        ]
        return {f"s{i}": pd.DataFrame(r, index=index, columns=["t1", "t2", "t3"]) for i, r in enumerate(rows)}

    def test_helper_follows_r_rule(self):
        series = self.series()
        self.assertEqual(bio.differentially_expressed_probesets(series, 0.7, 0.7, 4), ["up", "down"])
        self.assertEqual(bio.differentially_expressed_probesets(series, 0.7, 0.7, 2), ["up", "down", "mixed"])
        self.assertEqual(bio.differentially_expressed_probesets(series, 1.5, 1.5, 2), [])
        self.assertEqual(bio.differentially_expressed_probesets(series, 0.7, 1.5, 4), ["up"])
        self.assertEqual(bio.differentially_expressed_probesets(list(series.values()), 0.7, 0.7, 4), ["up", "down"])
        with self.assertRaises(ValueError):
            bio.differentially_expressed_probesets([f.iloc[:, :1] for f in series.values()])

    def test_result_structure_and_comparisons(self):
        mappings = {"up": ["UPG"], "down": ["DOWNG"]}
        result = bio.identify_differentially_expressed_genes(
            self.series(), cutoff_induction=0.7, cutoff_repression=0.7, majority=4,
            probeset_gene_name_mappings=mappings,
        )
        comparisons = result["DifferentialExpression"]
        self.assertEqual(list(comparisons), ["2To1", "3To1", "3To2"])
        late = comparisons["3To1"]
        self.assertEqual(late["Induced_ProbeID"], ["up"])
        self.assertEqual(late["Repressed_ProbeID"], ["down"])
        self.assertEqual(late["Induced_Genes"], {"up": ["UPG"]})
        self.assertEqual(late["Repressed_Genes"], {"down": ["DOWNG"]})
        self.assertEqual(late["Induced_M_Value"].shape, (1, 4))  # one probe set x 4 samples
        self.assertEqual(late["Induced_M_Value"].to_numpy().tolist(), [[1.0] * 4])
        self.assertEqual(comparisons["2To1"]["Induced_ProbeID"], [])  # nothing changes from t1 to t2
        self.assertEqual(list(result["DifferentialMappings"]), ["up", "down"])

    def test_four_timepoints_give_six_comparisons_in_r_order(self):
        frame = pd.DataFrame(np.zeros((1, 4)), index=["g"])
        result = bio.identify_differentially_expressed_genes([frame, frame], majority=1)
        self.assertEqual(list(result["DifferentialExpression"]), ["2To1", "3To1", "3To2", "4To1", "4To2", "4To3"])

    def test_log2_scale_uses_ratios_and_rounds(self):
        frame = pd.DataFrame([[1.0, 4.0]], index=["g"])
        result = bio.identify_differentially_expressed_genes(
            [frame] * 2, cutoff_induction=2, majority=2, need_log2_scale=True, probeset_gene_name_mappings={},
        )
        entry = result["DifferentialExpression"]["2To1"]
        self.assertEqual(entry["Induced_ProbeID"], ["g"])
        self.assertEqual(entry["Induced_M_Value"].to_numpy().tolist(), [[2.0, 2.0]])

    def test_default_mapping_comes_from_david_and_single_timepoint_is_rejected(self):
        from fbnnet_core.datasets import load_dataset

        david = load_dataset("DAVID_Gene_List")
        probeset, symbol = david.iloc[0]["Probeset"], david.iloc[0]["Symbol"]
        frame = pd.DataFrame([[0.0, 3.0]], index=[probeset])
        result = bio.identify_differentially_expressed_genes([frame], majority=1)
        self.assertEqual(result["DifferentialMappings"], {probeset: [symbol]})
        with self.assertRaises(ValueError):
            bio.identify_differentially_expressed_genes([frame.iloc[:, :1]])

    def test_writes_m_value_files_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            bio.identify_differentially_expressed_genes(
                self.series(), majority=4, probeset_gene_name_mappings={}, output_dir=folder, name_tab="RMA",
            )
            written = sorted(p.name for p in Path(folder).iterdir())
        self.assertEqual(written, ["MValue_2To1_RMA.csv", "MValue_3To1_RMA.csv", "MValue_3To2_RMA.csv"])


class TestGeneLevelSeries(unittest.TestCase):
    def setUp(self):
        index = ["p1", "p2", "p3", "p9"]
        self.samples = {
            "s1": pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [5.0, 5.0]], index=index),
            "s2": pd.DataFrame([[2.0, 3.0], [4.0, 5.0], [30.0, 40.0], [6.0, 6.0]], index=index),
        }

    def test_get_specific_expressed_genes_renames_and_keeps_common_genes(self):
        result = bio.get_specific_expressed_genes(self.samples, annotation=ANNOTATION)
        known = result["KnownDiffExpressedTimeSeriesData"]
        # p9 has no annotation; p1 and p2 share the symbol AAA, the first probe set is kept
        self.assertEqual(sorted(known["s1"].index), ["AAA", "BBB"])
        self.assertEqual(known["s1"].loc["AAA"].tolist(), [1.0, 2.0])
        self.assertEqual(result["UnKnownDiffExpressedTimeSeriesData"], {})
        self.assertEqual(list(result["AllDiffExpressedTimeSeriesData"]), ["s1", "s2"])
        self.assertIsInstance(result["ProbesetMapping"], bio.AnnotationMappedObject)

    def test_common_genes_must_exist_in_every_sample(self):
        samples = dict(self.samples)
        samples["s2"] = samples["s2"].drop(index=["p3"])
        known = bio.get_specific_expressed_genes(samples, annotation=ANNOTATION)["KnownDiffExpressedTimeSeriesData"]
        self.assertEqual(list(known["s1"].index), ["AAA"])
        self.assertEqual(list(known["s2"].index), ["AAA"])

    def test_genelist_accepts_probeset_ids_and_symbols(self):
        by_probeset = bio.get_specific_expressed_genes(self.samples, ["p3"], ANNOTATION)
        self.assertEqual(list(by_probeset["KnownDiffExpressedTimeSeriesData"]["s1"].index), ["BBB"])
        by_symbol = bio.get_specific_expressed_genes(self.samples, ["AAA"], ANNOTATION)
        self.assertEqual(list(by_symbol["KnownDiffExpressedTimeSeriesData"]["s2"].index), ["AAA"])

    def test_convert_timeseries_probeset_name_to_gene_name_averages_and_drops_unmapped(self):
        result = bio.convert_timeseries_probeset_name_to_gene_name(self.samples, mapping={"p1": "G", "p2": "G", "p3": "H"})
        data = result["convert_data"]
        self.assertEqual(list(data["s1"].index), ["G", "H"])  # p9 is unmapped and dropped
        self.assertEqual(data["s1"].loc["G"].tolist(), [2.0, 3.0])
        self.assertEqual(data["s2"].loc["H"].tolist(), [30.0, 40.0])
        self.assertEqual(result["alldifferExpressednames"], ["p1", "p2", "p3", "p9"])
        self.assertIsNone(result["ProbesetMapping"])

    def test_multiple_symbols_contribute_to_each_gene(self):
        mapping = pd.DataFrame({"Probeset": ["p1", "p1", "p2"], "Symbol": ["A", "B", "B"]})
        data = bio.convert_timeseries_probeset_name_to_gene_name(self.samples, mapping=mapping)["convert_data"]["s1"]
        self.assertEqual(data.loc["A"].tolist(), [1.0, 2.0])
        self.assertEqual(data.loc["B"].tolist(), [2.0, 3.0])  # mean of p1 and p2

    def test_annotation_deduplicates_by_entrez_id(self):
        result = bio.convert_timeseries_probeset_name_to_gene_name(self.samples, annotation=ANNOTATION)
        self.assertEqual(list(result["convert_data"]["s1"].index), ["AAA", "BBB"])
        self.assertEqual(result["mappingsheet"].columns.tolist(), ["original", "mapTo"])


class TestRawDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.cdf = root / "chip.cdf"
        write_cdf(cls.cdf)
        cls.cel_dir = root / "cel"
        cls.cel_dir.mkdir()
        rng = np.random.default_rng(5)
        for patient in ("B-ALL-1", "B-ALL-2"):
            for hour in (0, 8, 24):
                write_cel(cls.cel_dir / f"{patient}-{hour}h.CEL", rng.uniform(100, 5000, 16))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_get_related_affy_raw_data(self):
        raw = bio.get_related_affy_raw_data(self.cel_dir, self.cdf, workers=1)
        self.assertEqual(raw["SampleName"][0], "B-ALL-1-0h.CEL")
        self.assertEqual(raw["Probes"], ["AFFX-control_at", "G1_at", "G2_at", "G3_at"])
        self.assertEqual(raw["ExpressionData"].shape, (7, 6))  # 7 PM cells x 6 arrays
        self.assertEqual(raw["AffyBatchObject"].probe_counts.tolist(), [1, 2, 2, 2])
        with self.assertRaises(ValueError):
            bio.get_related_affy_raw_data(self.cel_dir.parent, self.cdf)

    def test_normalize_matches_cel_import_and_rejects_other_methods(self):
        raw = bio.get_related_affy_raw_data(self.cel_dir, self.cdf, workers=1)
        normalised = bio.normalize_timeseries_raw_data(raw)
        expected = cel_import.summarise_expression(sorted(self.cel_dir.glob("*.CEL")), self.cdf, workers=1)
        pd.testing.assert_frame_equal(normalised["NormalizedExpData"], expected)
        self.assertEqual(normalised["FeatureNames"], raw["Probes"])
        for method in ("GCRMA", "MAS5"):
            with self.assertRaises(NotImplementedError):
                bio.normalize_timeseries_raw_data(raw, method)

    def test_convert_affy_raw_data_matches_the_cel_pipeline_and_caches(self):
        with tempfile.TemporaryDirectory() as output:
            result = bio.convert_affy_raw_data_into_normalized_structure_data(
                self.cel_dir, self.cdf, output_dir=output, workers=1
            )
            self.assertEqual(list(result), ["B-ALL-1", "B-ALL-2"])
            self.assertEqual(list(result["B-ALL-1"].columns), [0, 8, 24])
            reference = cel_import.cel_to_timeseries(self.cel_dir, self.cdf, workers=1, binarise=False, exclude_prefixes=())
            np.testing.assert_allclose(result["B-ALL-1"].sort_index().to_numpy(), reference[0].sort_index().to_numpy())
            cached = sorted(p.name for p in Path(output).iterdir())
            self.assertEqual(cached, ["NormalizedEValue_RMA_cel.csv"])
            again = bio.convert_affy_raw_data_into_normalized_structure_data(
                [self.cel_dir], Path("missing.cdf"), output_dir=output, is_new=False
            )
            np.testing.assert_allclose(again["B-ALL-2"].to_numpy(), result["B-ALL-2"].to_numpy())
        with self.assertRaises(NotImplementedError):
            bio.convert_affy_raw_data_into_normalized_structure_data(self.cel_dir, self.cdf, use_gcrma=True)

    def test_several_directories_are_combined(self):
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        for hour in (0, 8, 24):
            write_cel(other / f"T-ALL-3-{hour}h.CEL", np.random.default_rng(hour).uniform(100, 5000, 16))
        combined = bio.convert_affy_raw_data_into_normalized_structure_data([self.cel_dir, other], self.cdf, workers=1)
        self.assertEqual(list(combined), ["B-ALL-1", "B-ALL-2", "T-ALL-3"])


REAL_CHIP_DB = Path(__file__).resolve().parents[1] / "data" / "testdata" / "HG-U133_Plus_2" / "hgu133plus2.sqlite"


def write_annotation_databases(folder: Path) -> Path:
    """Tiny hgu133plus2.sqlite + org.Hs.eg.sqlite with the schema of the Bioconductor packages."""
    chip = folder / "hgu133plus2.sqlite"
    with contextlib.closing(sqlite3.connect(chip)) as db, db:  # closed (Windows cannot delete open files)
        db.execute("CREATE TABLE probes (probe_id VARCHAR(80), gene_id VARCHAR(10), is_multiple SMALLINT)")
        db.executemany("INSERT INTO probes VALUES (?, ?, ?)", [
            ("p1", "100", 0), ("p2", "100", 0), ("p3", "200", 1), ("p3", "300", 1), ("p4", None, 0), ("p5", "400", 0),
        ])
    with contextlib.closing(sqlite3.connect(folder / "org.Hs.eg.sqlite")) as db, db:
        db.execute("CREATE TABLE genes (_id INTEGER PRIMARY KEY, gene_id VARCHAR(10))")
        db.execute("CREATE TABLE gene_info (_id INTEGER, gene_name VARCHAR(255), symbol VARCHAR(80))")
        db.execute("CREATE TABLE unigene (_id INTEGER, unigene_id VARCHAR(20))")
        db.executemany("INSERT INTO genes VALUES (?, ?)", [(1, "100"), (2, "200"), (3, "300"), (4, "400")])
        db.executemany("INSERT INTO gene_info VALUES (?, ?, ?)", [
            (1, "gene one", "ONE"), (2, "gene two", "TWO"), (3, "gene three", "THREE"), (4, "gene four", "FOUR"),
        ])
        db.executemany("INSERT INTO unigene VALUES (?, ?)", [(1, "Hs.1"), (2, "Hs.2")])
    return chip


class TestHgu133plus2Database(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.chip = write_annotation_databases(self.folder)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_returns_one_row_per_probe_and_gene_in_database_order(self):
        table = bio.load_hgu133plus2_annotation(self.chip)
        self.assertEqual(list(table.columns), ["PROBEID", "SYMBOL", "ENTREZID", "GENENAME", "UNIGENE"])
        self.assertEqual(table["PROBEID"].tolist(), ["p1", "p2", "p3", "p3", "p4", "p5"])
        self.assertEqual(table["SYMBOL"].dropna().tolist(), ["ONE", "ONE", "TWO", "THREE", "FOUR"])
        self.assertTrue(pd.isna(table.loc[4, "SYMBOL"]))
        self.assertEqual(table.loc[2, "ENTREZID"], "200")
        self.assertEqual(table.loc[0, "GENENAME"], "gene one")
        self.assertEqual(table.loc[0, "UNIGENE"], "Hs.1")
        self.assertTrue(pd.isna(table.loc[5, "UNIGENE"]))

    def test_recent_org_database_without_unigene_table(self):
        with contextlib.closing(sqlite3.connect(self.folder / "org.Hs.eg.sqlite")) as db, db:
            db.execute("DROP TABLE unigene")
        table = bio.load_hgu133plus2_annotation(self.chip)
        self.assertEqual(table["SYMBOL"].dropna().tolist(), ["ONE", "ONE", "TWO", "THREE", "FOUR"])
        self.assertTrue(table["UNIGENE"].isna().all())

    def test_explicit_gene_database_and_missing_files(self):
        elsewhere = self.folder / "elsewhere"
        elsewhere.mkdir()
        moved = elsewhere / "org.db"
        (self.folder / "org.Hs.eg.sqlite").rename(moved)
        self.assertEqual(len(bio.load_hgu133plus2_annotation(self.chip, moved)), 6)
        with self.assertRaisesRegex(FileNotFoundError, "org.Hs.eg.sqlite"):
            bio.load_hgu133plus2_annotation(self.chip)
        with self.assertRaises(FileNotFoundError):
            bio.load_hgu133plus2_annotation(self.folder / "nope.sqlite")

    def test_invalid_database_is_reported(self):
        bad = self.folder / "bad"
        bad.mkdir()
        (bad / "hgu133plus2.sqlite").write_bytes(b"not a database")
        (bad / "org.Hs.eg.sqlite").write_bytes(b"not a database")
        with self.assertRaises(ValueError):
            bio.load_hgu133plus2_annotation(bad / "hgu133plus2.sqlite")

    def test_path_is_accepted_wherever_an_annotation_is(self):
        mapped = bio.map_to_gene_name_with_hgu133plus2_db(["p1", "p2", "p3", "p4"], self.chip)
        # p3 keeps its first gene (TWO), p4 has no gene; p1 and p2 share Entrez 100
        self.assertEqual(sorted(mapped.MappedProbeset["PROBEID"]), ["p1", "p2", "p3"])
        self.assertEqual(mapped.DeDuplicatedMappedProbeset["SYMBOL"].tolist(), ["ONE", "TWO"])
        names = bio.map_probeset_names(["p1", "p2", "p3", "p4"], str(self.chip))
        self.assertEqual(names, {"p1": ["ONE"], "p2": ["ONE"], "p3": ["TWO"], "p4": ["N/A"]})

    def test_environment_variable_sets_the_default_annotation(self):
        with mock.patch.dict(os.environ, {bio.ANNOTATION_ENV: str(self.chip)}):
            self.assertEqual(bio.map_probeset_names(["p5"]), {"p5": ["FOUR"]})
            mapped = bio.map_to_gene_name_with_hgu133plus2_db(["p5"])
            self.assertEqual(mapped.MappedProbeset["SYMBOL"].tolist(), ["FOUR"])
        self.assertEqual(bio.map_probeset_names(["p5"]), {"p5": ["N/A"]})  # back to the DAVID list

    def test_gene_level_series_use_the_database(self):
        samples = {
            "s1": pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [7.0, 7.0]], index=["p1", "p2", "p3", "p4"]),
            "s2": pd.DataFrame([[2.0, 3.0], [4.0, 5.0], [30.0, 40.0], [8.0, 8.0]], index=["p1", "p2", "p3", "p4"]),
        }
        converted = bio.convert_timeseries_probeset_name_to_gene_name(samples, annotation=self.chip)
        self.assertEqual(list(converted["convert_data"]["s1"].index), ["ONE", "TWO"])
        self.assertEqual(converted["convert_data"]["s1"].loc["ONE"].tolist(), [2.0, 3.0])
        specific = bio.get_specific_expressed_genes(samples, annotation=self.chip)
        self.assertEqual(sorted(specific["KnownDiffExpressedTimeSeriesData"]["s1"].index), ["ONE", "TWO"])
        cel_style = bio.convert_timeseries_probeset_name_to_gene_name(samples, mapping=self.chip)
        self.assertEqual(list(cel_style["convert_data"]["s2"].index), ["ONE", "TWO"])

    @unittest.skipUnless(REAL_CHIP_DB.exists(), "hgu133plus2.sqlite is not in data/testdata/HG-U133_Plus_2")
    def test_real_database(self):
        table = bio.load_hgu133plus2_annotation(REAL_CHIP_DB)
        self.assertGreater(table["PROBEID"].nunique(), 54000)
        first = table[table["PROBEID"] == "1007_s_at"].iloc[0]
        self.assertEqual((first["SYMBOL"], first["ENTREZID"]), ("DDR1", "780"))
        self.assertEqual(bio.map_probeset_names(["1053_at"], REAL_CHIP_DB), {"1053_at": ["RFC2"]})


class TestExports(unittest.TestCase):
    def test_functions_are_available_from_the_package(self):
        import fbnnet_core

        for name in (
            "convert_affy_raw_data_into_normalized_structure_data", "get_related_affy_raw_data",
            "normalize_timeseries_raw_data", "convert_into_sample_timeseries", "reorder_sample_timeseries",
            "identify_differentially_expressed_genes", "get_specific_expressed_genes",
            "convert_timeseries_probeset_name_to_gene_name", "map_probeset_names",
            "map_to_gene_name_with_hgu133plus2_db",
        ):
            self.assertTrue(callable(getattr(fbnnet_core, name)), name)


if __name__ == "__main__":
    unittest.main()
