"""GEO download helpers and probe-set name mapping (network access is mocked)."""
import gzip
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from fbnnet_core import geo

MATRIX = (
    '!Series_title\t"Demo"\n'
    '!Sample_title\t"time 0 subject 1"\t"time 8 subject 1"\n'
    '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
    "!series_matrix_table_begin\n"
    '"ID_REF"\t"GSM1"\t"GSM2"\n'
    '"p1"\t1.5\t2.5\n'
    '"p2"\t3\tnull\n'
    "!series_matrix_table_end\n"
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fake_open_factory(routes):
    def fake_open(url, timeout=60.0):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return FakeResponse(payload)
        raise RuntimeError(f"unexpected url {url}")
    return fake_open


class TestGeo(unittest.TestCase):
    def test_series_url(self):
        self.assertEqual(
            geo.series_url("GSE35635", "matrix"),
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE35nnn/GSE35635/matrix/",
        )
        self.assertEqual(geo.series_url("gse2677", "suppl"), "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2677/suppl/")
        self.assertEqual(geo.series_url("GSE12", "matrix"), "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE12/matrix/")
        for bad in ("GPL570", "35635", "GSE", "GSE12a"):
            with self.assertRaises(ValueError):
                geo.series_url(bad, "matrix")

    def test_parse_series_matrix(self):
        exprs = geo.parse_series_matrix(MATRIX.splitlines(keepends=True))
        self.assertEqual(list(exprs.index), ["p1", "p2"])
        self.assertEqual(list(exprs.columns), ["GSM1", "GSM2"])
        self.assertEqual(exprs.loc["p1"].tolist(), [1.5, 2.5])
        self.assertTrue(pd.isna(exprs.loc["p2", "GSM2"]))
        self.assertEqual(exprs.attrs["titles"], {"GSM1": "time 0 subject 1", "GSM2": "time 8 subject 1"})
        with self.assertRaises(ValueError):
            geo.parse_series_matrix(["!Series_title\tx\n"])

    def test_download_genome_data_exprs_picks_platform_and_caches(self):
        listing = (
            b'<a href="GSE9_series_matrix.txt.gz">a</a>'
            b'<a href="GSE9-GPL570_series_matrix.txt.gz">b</a>'
            b'<a href="GSE9-GPL96_series_matrix.txt.gz">c</a><a href="/geo/">up</a>'
        )
        other = MATRIX.replace('"p1"', '"q1"')
        routes = {
            "/matrix/": listing,
            "GSE9-GPL96_series_matrix.txt.gz": gzip.compress(other.encode()),
            "GSE9-GPL570_series_matrix.txt.gz": gzip.compress(MATRIX.encode()),
            "GSE9_series_matrix.txt.gz": gzip.compress(MATRIX.encode()),
        }
        with tempfile.TemporaryDirectory() as cache, mock.patch.object(geo, "_open", fake_open_factory(routes)):
            first = geo.download_genome_data_exprs("GSE9")  # first file, like getGEO(...)[[1]]
            self.assertEqual(list(first.index), ["p1", "p2"])
            picked = geo.download_genome_data_exprs("GSE9", platform="gpl96", cache_dir=cache)
            self.assertEqual(list(picked.index), ["q1", "p2"])
            self.assertTrue((Path(cache) / "GSE9-GPL96_series_matrix.txt.gz").exists())
        with mock.patch.object(geo, "_open", fake_open_factory(routes)):
            with self.assertRaises(RuntimeError):
                geo.download_genome_data_exprs("GSE9", platform="GPL1")

    def test_cached_file_is_reused_without_network(self):
        listing = b'<a href="GSE9_series_matrix.txt.gz">a</a>'
        with tempfile.TemporaryDirectory() as cache:
            (Path(cache) / "GSE9_series_matrix.txt.gz").write_bytes(gzip.compress(MATRIX.encode()))
            with mock.patch.object(geo, "_open", fake_open_factory({"/matrix/": listing})):
                exprs = geo.download_genome_data_exprs("GSE9", cache_dir=cache)
        self.assertEqual(exprs.shape, (2, 2))

    def test_download_geo_raw_files_extracts_and_gunzips(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, data in (("S1-0h.CEL.gz", gzip.compress(b"cel one")), ("notes.txt", b"hello")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            evil = tarfile.TarInfo("../escape.txt")
            evil.size = 1
            tar.addfile(evil, io.BytesIO(b"x"))
        with tempfile.TemporaryDirectory() as dest, mock.patch.object(
            geo, "_open", fake_open_factory({"GSE5_RAW.tar": buffer.getvalue()})
        ):
            files = geo.download_geo_raw_files("GSE5", Path(dest) / "raw")
            self.assertEqual([f.name for f in files], ["S1-0h.CEL", "notes.txt"])
            self.assertEqual((Path(dest) / "raw" / "S1-0h.CEL").read_bytes(), b"cel one")
            self.assertFalse((Path(dest) / "escape.txt").exists())
            self.assertFalse((Path(dest) / "raw" / "GSE5_RAW.tar").exists())  # archive removed
        with self.assertRaises(ValueError):
            geo.download_geo_raw_files("nope", "x")

    def test_download_failure_is_reported(self):
        with mock.patch.object(geo, "urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                geo.download_genome_data_exprs("GSE9")


def package_tarball(package, filename, payload):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in ((f"{package}/DESCRIPTION", b"Package: x"), (f"{package}/inst/extdata/{filename}", payload)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class TestDownloadAnnotation(unittest.TestCase):
    INDEX = (
        b"Package: hgu133plus2.db\nVersion: 3.13.0\nDepends: R\n\n"
        b"Package: hgu133plus2cdf\nVersion: 2.18.0\n\n"
        b"Package: org.Hs.eg.db\nVersion: 3.23.1\nDepends: R (>= 2.7.0),\n methods\n"
    )

    def routes(self):
        return {
            "/PACKAGES": self.INDEX,
            "hgu133plus2.db_3.13.0.tar.gz": package_tarball("hgu133plus2.db", "hgu133plus2.sqlite", b"chip db"),
            "org.Hs.eg.db_3.23.1.tar.gz": package_tarball("org.Hs.eg.db", "org.Hs.eg.sqlite", b"org db"),
        }

    def test_downloads_both_databases_and_keeps_existing_files(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(geo, "_open", fake_open_factory(self.routes())):
            files = geo.download_hgu133plus2_annotation(Path(folder) / "annotation")
            self.assertEqual([f.name for f in files], ["hgu133plus2.sqlite", "org.Hs.eg.sqlite"])
            self.assertEqual(files[0].read_bytes(), b"chip db")
            self.assertEqual(files[1].read_bytes(), b"org db")
            self.assertEqual([p.name for p in files[0].parent.iterdir() if p.name.endswith(".part")], [])
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "hgu133plus2.sqlite").write_bytes(b"mine")
            routes = self.routes()
            del routes["hgu133plus2.db_3.13.0.tar.gz"]  # must not be requested
            with mock.patch.object(geo, "_open", fake_open_factory(routes)):
                files = geo.download_hgu133plus2_annotation(folder)
            self.assertEqual(files[0].read_bytes(), b"mine")
            self.assertEqual(files[1].read_bytes(), b"org db")

    def test_nothing_is_requested_when_both_files_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("hgu133plus2.sqlite", "org.Hs.eg.sqlite"):
                (Path(folder) / name).write_bytes(b"x")
            with mock.patch.object(geo, "_open", side_effect=AssertionError("no network expected")):
                self.assertEqual(len(geo.download_hgu133plus2_annotation(folder)), 2)

    def test_overwrite_replaces_files(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(geo, "_open", fake_open_factory(self.routes())):
            (Path(folder) / "hgu133plus2.sqlite").write_bytes(b"old")
            (Path(folder) / "org.Hs.eg.sqlite").write_bytes(b"old")
            files = geo.download_hgu133plus2_annotation(folder, overwrite=True)
            self.assertEqual([f.read_bytes() for f in files], [b"chip db", b"org db"])

    def test_missing_package_or_member_is_reported(self):
        routes = self.routes()
        routes["/PACKAGES"] = b"Package: hgu133plus2.db\nVersion: 3.13.0\n"
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(geo, "_open", fake_open_factory(routes)):
            with self.assertRaisesRegex(RuntimeError, "org.Hs.eg.db"):
                geo.download_hgu133plus2_annotation(folder)
        routes = self.routes()
        routes["hgu133plus2.db_3.13.0.tar.gz"] = package_tarball("hgu133plus2.db", "other.sqlite", b"x")
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(geo, "_open", fake_open_factory(routes)):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                geo.download_hgu133plus2_annotation(folder)

    def test_downloaded_files_can_be_loaded(self):
        from test_bioinformatics import write_annotation_databases
        from fbnnet_core import bioinformatics

        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as folder:
            chip = write_annotation_databases(Path(source))
            routes = self.routes()
            routes["hgu133plus2.db_3.13.0.tar.gz"] = package_tarball("hgu133plus2.db", "hgu133plus2.sqlite", chip.read_bytes())
            routes["org.Hs.eg.db_3.23.1.tar.gz"] = package_tarball(
                "org.Hs.eg.db", "org.Hs.eg.sqlite", (Path(source) / "org.Hs.eg.sqlite").read_bytes()
            )
            with mock.patch.object(geo, "_open", fake_open_factory(routes)):
                files = geo.download_hgu133plus2_annotation(folder)
            table = bioinformatics.load_hgu133plus2_annotation(files[0])
            self.assertEqual(table["SYMBOL"].dropna().tolist(), ["ONE", "ONE", "TWO", "THREE", "FOUR"])


if __name__ == "__main__":
    unittest.main()
