# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Leshi Chen
"""Downloads of public data: NCBI GEO series and the Bioconductor probe set annotation.

Python counterparts of the R package's ``downloadGenomeDataExprs`` (which used
``GEOquery::getGEO(..., GSEMatrix = TRUE)``), plus a helper for the raw files of a
series. Only the standard library, NumPy and pandas are needed.

Example::

    from fbnnet_core.geo import download_genome_data_exprs
    exprs = download_genome_data_exprs("GSE35635")   # probes x samples (log2 expression)

The raw CEL files of a series (for :mod:`fbnnet_core.cel_import`) are large and are not
shipped with this package; fetch them with :func:`download_geo_raw_files`. The probe set
annotation of the HG-U133 Plus 2.0 array (``hgu133plus2.sqlite`` and ``org.Hs.eg.sqlite``, from
the Bioconductor packages ``hgu133plus2.db`` and ``org.Hs.eg.db``) is not shipped either; fetch it
with :func:`download_hgu133plus2_annotation`.
"""
from __future__ import annotations

import gzip
import io
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"
_ACCESSION = re.compile(r"^GSE(\d+)$", re.IGNORECASE)
PathLike = Union[str, "Path"]


def _open(url: str, timeout: float = 60.0):
    request = Request(url, headers={"User-Agent": "fbnnet-core"})
    try:
        return urlopen(request, timeout=timeout)
    except (URLError, OSError) as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error


def series_url(accession: str, section: str) -> str:
    """Directory URL of a series on the GEO FTP site; `section` is ``matrix`` or ``suppl``."""
    match = _ACCESSION.match(accession.strip())
    if not match:
        raise ValueError(f"Not a GEO series accession (expected e.g. 'GSE35635'): {accession!r}")
    digits = match.group(1)
    stub = f"GSE{digits[:-3]}nnn"
    return f"{GEO_FTP}/{stub}/GSE{digits}/{section}/"


def _listing(url: str) -> List[str]:
    with _open(url) as response:
        html = response.read().decode("utf-8", errors="replace")
    return re.findall(r'href="([^"/?]+)"', html)


def parse_series_matrix(lines: Iterable[str]) -> pd.DataFrame:
    """Expression table (probes x samples) from the text of a GEO series matrix file.

    The sample titles are kept in ``result.attrs["titles"]`` (GSM accession -> title).
    """
    table: List[str] = []
    accessions: List[str] = []
    titles: List[str] = []
    inside = False
    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            inside = True
        elif line.startswith("!series_matrix_table_end"):
            break
        elif inside:
            table.append(line)
        elif line.startswith("!Sample_geo_accession"):
            accessions = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_title"):
            titles = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
    if not table:
        raise ValueError("No expression table found in the series matrix file")
    exprs = pd.read_csv(io.StringIO("".join(table)), sep="\t", index_col=0, quotechar='"', na_values=["null", ""])
    exprs.index.name = None
    exprs.attrs["titles"] = dict(zip(accessions, titles))
    return exprs


def download_genome_data_exprs(
    accession: str,
    platform: Optional[str] = None,
    cache_dir: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Normalised expression matrix of a GEO series (R ``downloadGenomeDataExprs``).

    Args:
        accession: Series accession such as ``"GSE35635"``.
        platform: Platform accession such as ``"GPL570"``. Series measured on several
            platforms have one matrix per platform; the default is the first, like
            ``getGEO(...)[[1]]`` in R.
        cache_dir: Optional directory where the downloaded ``.txt.gz`` is kept and reused.

    Returns:
        Probes as rows and samples (GSM accessions) as columns, with the sample titles in
        ``attrs["titles"]``.
    """
    base = series_url(accession, "matrix")
    names = sorted(n for n in _listing(base) if n.endswith("_series_matrix.txt.gz"))
    if platform:
        names = [n for n in names if platform.upper() in n.upper()]
    if not names:
        raise RuntimeError(f"No series matrix file found for {accession}" + (f" on {platform}" if platform else ""))
    name = names[0]
    logger.info("Downloading %s", base + name)

    local = Path(cache_dir) / name if cache_dir is not None else None
    if local is not None and local.exists():
        payload = local.read_bytes()
    else:
        with _open(base + name) as response:
            payload = response.read()
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(payload)
    with gzip.open(io.BytesIO(payload), "rt", errors="replace") as handle:
        return parse_series_matrix(handle)


def download_geo_raw_files(accession: str, dest_dir: PathLike, extract: bool = True) -> List[Path]:
    """Downloads ``<accession>_RAW.tar`` and unpacks it, returning the files in `dest_dir`.

    Affymetrix series contain ``.CEL.gz`` files; they are decompressed to ``.CEL`` so that
    :func:`fbnnet_core.cel_import.cel_to_timeseries` can read them. The archives are large
    (GSE2677 is about 1 GB), and other platforms (e.g. Agilent) contain other file types.
    """
    match = _ACCESSION.match(accession.strip())
    if not match:
        raise ValueError(f"Not a GEO series accession (expected e.g. 'GSE2677'): {accession!r}")
    name = f"GSE{match.group(1)}_RAW.tar"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / name
    url = series_url(accession, "suppl") + name
    logger.info("Downloading %s", url)
    with _open(url, timeout=600.0) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if not extract:
        return [archive]

    files: List[Path] = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not member.isfile() or dest.resolve() not in target.parents:
                continue  # skip directories and anything that would escape `dest`
            source = tar.extractfile(member)
            if member.name.lower().endswith(".gz"):
                target = target.with_suffix("")
                with gzip.open(source) as unpacked, open(target, "wb") as out:
                    shutil.copyfileobj(unpacked, out, length=1 << 20)
            else:
                with open(target, "wb") as out:
                    shutil.copyfileobj(source, out, length=1 << 20)
            files.append(target)
    archive.unlink()
    return sorted(files, key=lambda path: path.name)  # Path ordering is case-insensitive on Windows only


BIOCONDUCTOR_ANNOTATION = "https://bioconductor.org/packages/release/data/annotation/src/contrib/"
ANNOTATION_PACKAGES = {
    "hgu133plus2.db": "hgu133plus2.sqlite",
    "org.Hs.eg.db": "org.Hs.eg.sqlite",
}


def _latest_versions(packages: Iterable[str]) -> dict:
    with _open(BIOCONDUCTOR_ANNOTATION + "PACKAGES") as response:
        index = response.read().decode("utf-8", errors="replace")
    versions = {}
    for block in index.split("\n\n"):
        fields = dict(re.findall(r"^(Package|Version):\s*(\S+)", block, re.MULTILINE))
        if fields.get("Package") in packages and "Version" in fields:
            versions[fields["Package"]] = fields["Version"]
    missing = [p for p in packages if p not in versions]
    if missing:
        raise RuntimeError(f"Bioconductor does not list {', '.join(missing)}")
    return versions


def download_hgu133plus2_annotation(dest_dir: PathLike, overwrite: bool = False) -> List[Path]:
    """Downloads the SQLite annotation of the HG-U133 Plus 2.0 array from Bioconductor.

    The two files are the databases inside the (R source) packages ``hgu133plus2.db`` (probe set ->
    Entrez gene, about 2 MB download) and ``org.Hs.eg.db`` (Entrez gene -> symbol and name, about
    100 MB download, 400 MB on disk). No R and no conversion are needed: the packages ship
    ready-made SQLite files, which are extracted into `dest_dir`. Read them with
    :func:`fbnnet_core.load_hgu133plus2_annotation` or pass the path of ``hgu133plus2.sqlite``
    wherever an annotation is accepted. The data are licensed under Artistic-2.0 by their authors.

    Returns the paths of the two files; files that already exist are kept unless `overwrite`.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {pkg: dest / name for pkg, name in ANNOTATION_PACKAGES.items()}
    todo = [pkg for pkg, path in wanted.items() if overwrite or not path.exists()]
    if todo:
        versions = _latest_versions(todo)
        for package in todo:
            url = f"{BIOCONDUCTOR_ANNOTATION}{package}_{versions[package]}.tar.gz"
            logger.info("Downloading %s", url)
            member_name = f"{package}/inst/extdata/{ANNOTATION_PACKAGES[package]}"
            found = False
            with _open(url, timeout=600.0) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:
                    if member.name == member_name:
                        partial = wanted[package].with_name(wanted[package].name + ".part")
                        with open(partial, "wb") as out:
                            shutil.copyfileobj(tar.extractfile(member), out, length=1 << 20)
                        partial.replace(wanted[package])
                        found = True
                        break
            if not found:
                raise RuntimeError(f"{member_name} not found in {url}")
    return list(wanted.values())
