# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Leshi Chen
"""Bioinformatics tools of the R package (``R/FBNBioinformatics.R`` and ``R/sample_FBN.R``).

They turn normalised expression data into the time series that FBNNet mines, and cover
the whole path from raw CEL files to gene-level data:

==========================================  ============================================================
Python                                      R
==========================================  ============================================================
``convert_affy_raw_data_into_normalized_structure_data``  ``convertAffyRawDataIntoNormalizedStructureData``
``get_related_affy_raw_data``               ``getRelatedAffyRawData``
``normalize_timeseries_raw_data``           ``normalizeTimesereisRawData``
``convert_into_sample_timeseries``          ``convertIntoSampleTimeSeries``
``reorder_sample_timeseries``               ``reorderSampleTimeSeries``
``identify_differentially_expressed_genes`` ``identifyDifferentiallyExpressedGenes``
``get_specific_expressed_genes``            ``getSpecificExpressedGenes``
``convert_timeseries_probeset_name_to_gene_name``  ``convertTimeseriesProbsetNameToGeneName``
``map_probeset_names``                      ``mapProbesetNames``
``map_to_gene_name_with_hgu133plus2_db``    ``mapToGeneNameWithhgu133plus2Db``
``convert_probeset_to_gene``                ``convert_probeset_to_gene``
``dissolve``                                ``dissolve``
(see :mod:`fbnnet_core.geo`)                ``downloadGenomeDataExprs``
==========================================  ============================================================

R lists of matrices become ``dict`` objects of DataFrames (sample name -> genes x timepoints);
a plain list of DataFrames is accepted wherever a dict is expected. Probe set annotation:
R reads the ``hgu133plus2.db`` Bioconductor package. Its SQLite file is read directly with
:func:`load_hgu133plus2_annotation`; wherever an annotation is accepted you can pass the path of
``hgu133plus2.sqlite`` (next to ``org.Hs.eg.sqlite``, which holds the symbols and gene names), a
DataFrame with the columns ``PROBEID``, ``SYMBOL``, ``ENTREZID`` and ``GENENAME`` (what
``AnnotationDbi::select`` returns), or nothing. With nothing, the path in the environment variable
``FBNNET_HGU133PLUS2_DB`` is used when set, otherwise the bundled ``DAVID_Gene_List``, which R
also uses as its fallback.

Differences from R are noted in the docstrings: GCRMA and MAS5 are not implemented (RMA runs
without its background-correction step), and the results are returned instead of being
written to an ``output`` folder unless ``output_dir`` is given.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, "Path"]
SampleSeries = Union[Dict[str, pd.DataFrame], Sequence[pd.DataFrame]]


def _as_samples(series: SampleSeries) -> Dict[str, pd.DataFrame]:
    if isinstance(series, dict):
        return dict(series)
    return {f"sample{i + 1}": frame for i, frame in enumerate(series)}


# --------------------------------------------------------------------------- sample grouping

def convert_into_sample_timeseries(
    normalized_data: pd.DataFrame,
    func: Optional[Callable[[List[str]], str]] = None,
    splitor: str = "-",
) -> Dict[str, pd.DataFrame]:
    """Groups the arrays (columns) of a normalised matrix into samples (R ``convertIntoSampleTimeSeries``).

    The column names are split at `splitor` and `func` maps the pieces to the sample name
    (default: the first three pieces joined with ``-``, so ``B-ALL-13-8h.CEL`` -> ``B-ALL-13``).
    """
    func = func or (lambda parts: "-".join(parts[:3]))
    groups = [func(str(name).split(splitor)) for name in normalized_data.columns]
    result: Dict[str, pd.DataFrame] = {}
    for group in dict.fromkeys(groups):
        result[group] = normalized_data.loc[:, [g == group for g in groups]]
    return result


def reorder_sample_timeseries(
    converted: SampleSeries,
    func: Optional[Callable[[List[str]], str]] = None,
    splitor: str = "-",
) -> Dict[str, pd.DataFrame]:
    """Orders every sample's columns by time and its rows by name (R ``reorderSampleTimeSeries``).

    The time of an array is the number in the last piece (default `func`) of its name
    after the first ``.``-suffix is removed (``B-ALL-13-8h.CEL`` -> ``8``). The columns are
    relabelled with these integers.
    """
    func = func or (lambda parts: parts[-1])
    result: Dict[str, pd.DataFrame] = {}
    for name, frame in _as_samples(converted).items():
        times = []
        for column in frame.columns:
            piece = func(re.sub(r"\..*", "", str(column)).split(splitor))
            numbers = re.findall(r"[0-9]+", piece)
            if not numbers:
                raise ValueError(f"No time found in column name {column!r}")
            times.append(int(numbers[0]))
        ordered = frame.copy()
        ordered.columns = times
        ordered = ordered.iloc[:, np.argsort(times, kind="stable")]
        result[name] = ordered.loc[sorted(ordered.index)]
    return result


def dissolve(x: Any, name: Optional[str] = None) -> Dict[str, Any]:
    """Flattens nested lists and dicts into one dict of their leaves (R ``dissolve``)."""
    flat: Dict[str, Any] = {}

    def walk(node: Any, label: Optional[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, str(key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{label}.{index + 1}" if label else str(index + 1))
        else:
            flat[label if label is not None else str(len(flat) + 1)] = node

    walk(x, name)
    return flat


# --------------------------------------------------------------------------- annotation

@dataclass
class AnnotationMappedObject:
    """Result of :func:`map_to_gene_name_with_hgu133plus2_db` (R class ``AnnotationMappedObject``)."""

    MappedProbeset: pd.DataFrame
    DeDuplicatedMappedProbeset: pd.DataFrame
    DeDuplicatedMapping: pd.DataFrame


_ANNOTATION_COLUMNS = ["PROBEID", "SYMBOL", "ENTREZID", "GENENAME"]


_ANNOTATION_CACHE: Dict[Any, pd.DataFrame] = {}
ORG_HS_EG_DB = "org.Hs.eg.sqlite"
ANNOTATION_ENV = "FBNNET_HGU133PLUS2_DB"


def load_hgu133plus2_annotation(chip_db: PathLike, gene_db: Optional[PathLike] = None) -> pd.DataFrame:
    """Probe set annotation from the SQLite files of ``hgu133plus2.db`` and ``org.Hs.eg.db``.

    Equivalent to ``AnnotationDbi::select(hgu133plus2.db, keys = <all probe sets>, columns =
    c("UNIGENE", "SYMBOL", "ENTREZID", "GENENAME"), keytype = "PROBEID")``: one row per probe set
    and Entrez gene, in database order (probe sets measuring several genes appear several times),
    with missing annotation as NaN (``UNIGENE`` is always NaN with recent ``org.Hs.eg.db``
    releases, which dropped the retired UniGene table).

    Args:
        chip_db: ``hgu133plus2.sqlite`` (Bioconductor package ``hgu133plus2.db``, folder ``extdata``).
        gene_db: ``org.Hs.eg.sqlite`` (package ``org.Hs.eg.db``). Default: the file of that name next
            to `chip_db`.

    Returns:
        DataFrame with the columns ``PROBEID``, ``SYMBOL``, ``ENTREZID``, ``GENENAME``, ``UNIGENE``.
    """
    chip = Path(chip_db)
    gene = Path(gene_db) if gene_db is not None else chip.with_name(ORG_HS_EG_DB)
    for path in (chip, gene):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; get hgu133plus2.sqlite and {ORG_HS_EG_DB} with "
                "fbnnet_core.download_hgu133plus2_annotation(folder), or copy them from the R packages "
                "hgu133plus2.db and org.Hs.eg.db (folder extdata) into one directory"
            )
    key = (str(chip.resolve()), chip.stat().st_mtime, str(gene.resolve()), gene.stat().st_mtime)
    if key not in _ANNOTATION_CACHE:
        connection = sqlite3.connect(f"file:{chip.resolve().as_posix()}?mode=ro", uri=True)
        try:
            connection.execute("ATTACH DATABASE ? AS org", (f"file:{gene.resolve().as_posix()}?mode=ro",))
            has_unigene = connection.execute(
                "SELECT 1 FROM org.sqlite_master WHERE type = 'table' AND name = 'unigene'"
            ).fetchone() is not None
            unigene_column, unigene_join = (
                ("u.unigene_id", "LEFT JOIN org.unigene u ON u._id = g._id ") if has_unigene else ("NULL", "")
            )
            rows = connection.execute(
                f"SELECT p.probe_id, i.symbol, g.gene_id, i.gene_name, {unigene_column} "
                "FROM main.probes p "
                "LEFT JOIN org.genes g ON g.gene_id = p.gene_id "
                "LEFT JOIN org.gene_info i ON i._id = g._id "
                f"{unigene_join}"
                "ORDER BY p.rowid"
            ).fetchall()
        except sqlite3.Error as error:
            raise ValueError(f"{chip} / {gene} are not valid hgu133plus2.db / org.Hs.eg.db files: {error}") from error
        finally:
            connection.close()
        _ANNOTATION_CACHE[key] = pd.DataFrame(rows, columns=["PROBEID", "SYMBOL", "ENTREZID", "GENENAME", "UNIGENE"])
    return _ANNOTATION_CACHE[key].copy()


def _annotation_table(annotation: Any) -> pd.DataFrame:
    """A DataFrame, the path of hgu133plus2.sqlite, the environment default or the DAVID list."""
    if isinstance(annotation, pd.DataFrame):
        return annotation
    if annotation is None:
        annotation = os.environ.get(ANNOTATION_ENV)
    if annotation is None:
        return _default_annotation()
    return load_hgu133plus2_annotation(annotation)


def _default_annotation() -> pd.DataFrame:
    from .datasets import load_dataset

    david = load_dataset("DAVID_Gene_List")
    return pd.DataFrame({
        "PROBEID": david["Probeset"].astype(str),
        "SYMBOL": david["Symbol"],
        "ENTREZID": david["ENTREZ_Gene_ID"].astype(str),
        "GENENAME": david["Gene.Name"],
    })


def map_to_gene_name_with_hgu133plus2_db(
    probesets: Sequence[str],
    annotation: Optional[pd.DataFrame] = None,
) -> Optional[AnnotationMappedObject]:
    """Looks probe sets up in an annotation table (R ``mapToGeneNameWithhgu133plus2Db``).

    Probe sets without symbol, Entrez id or gene name are dropped (non-gene probe sets), then
    duplicates by probe set are removed, and probe sets that share an Entrez id are mapped to
    one representative probe set (the first after ordering by Entrez id). Returns None when
    nothing maps.

    Args:
        probesets: Probe set ids to look up.
        annotation: Path of ``hgu133plus2.sqlite`` (see :func:`load_hgu133plus2_annotation`) or a
            DataFrame with ``PROBEID``, ``SYMBOL``, ``ENTREZID`` and ``GENENAME``. Default: the file named
            by the environment variable ``FBNNET_HGU133PLUS2_DB`` if set, else the bundled DAVID list.
    """
    table = _annotation_table(annotation)
    missing = [c for c in _ANNOTATION_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"annotation lacks the columns {missing}")
    table = table[table["PROBEID"].isin(list(probesets))]
    table = table[table[["SYMBOL", "ENTREZID", "GENENAME"]].notna().all(axis=1)]
    if table.empty:
        return None
    table = table.assign(ENTREZID=table["ENTREZID"].astype(str))
    mapped = table.drop_duplicates("PROBEID").sort_values("ENTREZID", kind="stable")
    deduplicated = mapped.drop_duplicates("ENTREZID")
    representative = dict(zip(deduplicated["ENTREZID"], deduplicated["PROBEID"]))
    sheet = pd.DataFrame({
        "original": mapped["PROBEID"].to_numpy(),
        "mapTo": mapped["ENTREZID"].map(representative).to_numpy(),
    })
    return AnnotationMappedObject(
        MappedProbeset=mapped.reset_index(drop=True),
        DeDuplicatedMappedProbeset=deduplicated.reset_index(drop=True),
        DeDuplicatedMapping=sheet,
    )


def convert_probeset_to_gene(annotation_mapped_object: AnnotationMappedObject, probeset: str) -> pd.DataFrame:
    """Annotation rows (symbol, Entrez id, gene name) of a probe set (R ``convert_probeset_to_gene``).

    The R function reads the mapping columns as ``orginal``/``mapto`` (typos), which never match,
    so it always returns nothing there; this port implements the documented intent.
    """
    if not isinstance(annotation_mapped_object, AnnotationMappedObject):
        raise TypeError("annotation_mapped_object must be an AnnotationMappedObject")
    sheet = annotation_mapped_object.DeDuplicatedMapping
    targets = sheet.loc[sheet["original"] == probeset, "mapTo"]
    rows = annotation_mapped_object.DeDuplicatedMappedProbeset
    return rows[rows["PROBEID"].isin(targets)]


def _symbols_by_probeset(mapping: Any) -> Dict[str, List[str]]:
    if mapping is None:
        return {}
    if isinstance(mapping, AnnotationMappedObject):
        sheet = mapping.DeDuplicatedMapping
        rows = mapping.DeDuplicatedMappedProbeset
        symbol_of = dict(zip(rows["PROBEID"], rows["SYMBOL"]))
        return {o: [str(symbol_of[t])] for o, t in zip(sheet["original"], sheet["mapTo"]) if t in symbol_of}
    if isinstance(mapping, pd.DataFrame):
        column = "PROBEID" if "PROBEID" in mapping.columns else "Probeset"
        symbols = "SYMBOL" if "SYMBOL" in mapping.columns else "Symbol"
        pairs = mapping[[column, symbols]].dropna()
        grouped: Dict[str, List[str]] = {}
        for probeset, symbol in zip(pairs[column].astype(str), pairs[symbols].astype(str)):
            grouped.setdefault(probeset, []).append(symbol)
        return grouped
    result: Dict[str, List[str]] = {}
    for key, value in mapping.items():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result[str(key)] = [str(v) for v in values if pd.notna(v)]
    return result


def map_probeset_names(probesets: Optional[Sequence[str]], names_mapped: Any = None) -> Dict[str, List[str]]:
    """Gene symbols of probe sets (R ``mapProbesetNames``).

    Every probe set is looked up in `names_mapped` first and then in the bundled
    ``DAVID_Gene_List``. `names_mapped` is the result of
    :func:`map_to_gene_name_with_hgu133plus2_db`, the path of ``hgu133plus2.sqlite``, a DataFrame
    (``PROBEID``/``SYMBOL`` or ``Probeset``/``Symbol`` columns) or a dict/Series of probe set -> symbol.
    When None and the environment variable ``FBNNET_HGU133PLUS2_DB`` names that file, it is used. Probe sets found
    in neither get ``["N/A"]``. Returns ``{probeset: [symbols]}``; empty input gives ``{}``.
    """
    if probesets is None or len(probesets) == 0:
        return {}
    from .datasets import load_dataset

    if names_mapped is None and os.environ.get(ANNOTATION_ENV):
        names_mapped = os.environ[ANNOTATION_ENV]
    if isinstance(names_mapped, (str, os.PathLike)):
        names_mapped = map_to_gene_name_with_hgu133plus2_db(probesets, names_mapped)
    primary = _symbols_by_probeset(names_mapped)
    david = _symbols_by_probeset(load_dataset("DAVID_Gene_List"))
    return {p: primary.get(p) or david.get(p) or ["N/A"] for p in probesets}


# --------------------------------------------------------------------------- differential expression

def _comparison_names(n_timepoints: int) -> List[tuple]:
    """(name, later column, earlier column) in R's order: 2To1, 3To1, 3To2, 4To1, ..."""
    pairs = [("2To1", 1, 0)]
    for later in range(3, n_timepoints + 1):
        for earlier in range(1, later):
            pairs.append((f"{later}To{earlier}", later - 1, earlier - 1))
    return pairs


def _m_values(samples: Dict[str, pd.DataFrame], need_log2_scale: bool) -> Dict[str, pd.DataFrame]:
    """Per comparison, a probe set x sample matrix of log2 ratios (R's ``folddata``/``tempMat``)."""
    first = next(iter(samples.values()))
    if first.shape[1] < 2:
        raise ValueError("The input must have columns more than one")
    result = {}
    for name, later, earlier in _comparison_names(first.shape[1]):
        columns = {}
        for sample, frame in samples.items():
            late, early = frame.iloc[:, later].to_numpy(float), frame.iloc[:, earlier].to_numpy(float)
            columns[sample] = np.round(np.log2(late / early), 5) if need_log2_scale else late - early
        result[name] = pd.DataFrame(columns, index=first.index)
    return result


def _passing(matrix: pd.DataFrame, cutoff_induction: float, cutoff_repression: float, majority: int):
    values = np.round(matrix.to_numpy(float), 6)
    induced = (values >= cutoff_induction).sum(axis=1) >= majority
    repressed = (values <= -cutoff_repression).sum(axis=1) >= majority
    return induced, repressed


def differentially_expressed_probesets(
    series: SampleSeries,
    cutoff_induction: float = 0.7,
    cutoff_repression: float = 0.7,
    majority: int = 7,
) -> List[str]:
    """Probe sets that are induced or repressed in any comparison, in input row order.

    A convenience over :func:`identify_differentially_expressed_genes` that returns only the
    probe set ids (what ``FBNLeukaemiaStudy.R`` collects as ``finalProbesetSet``).
    """
    samples = _as_samples(series)
    selected = np.zeros(len(next(iter(samples.values()))), dtype=bool)
    for matrix in _m_values(samples, False).values():
        induced, repressed = _passing(matrix, cutoff_induction, cutoff_repression, majority)
        selected |= induced | repressed
    return next(iter(samples.values())).index[selected].tolist()


def identify_differentially_expressed_genes(
    order_sample_timeseries: SampleSeries,
    cutoff_induction: float = 1,
    cutoff_repression: float = 1,
    majority: int = 7,
    need_log2_scale: bool = False,
    probeset_gene_name_mappings: Optional[Dict[str, List[str]]] = None,
    name_tab: str = "RMA",
    output_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Finds induced and repressed probe sets between timepoints (R ``identifyDifferentiallyExpressedGenes``).

    For every pair of timepoints (``2To1``, ``3To1``, ``3To2``, ...) the log2 change is taken per
    sample (a difference when the data are already log2, as after RMA; a log2 ratio when
    `need_log2_scale`). A probe set is *induced* when the change is >= `cutoff_induction`
    in at least `majority` samples and *repressed* when it is <= -`cutoff_repression` in at least
    `majority` samples.

    Args:
        order_sample_timeseries: Samples (dict or list) of probe sets x timepoints, as returned by
            :func:`reorder_sample_timeseries`. All samples need the same rows and timepoints.
        probeset_gene_name_mappings: ``{probeset: [symbols]}``; default :func:`map_probeset_names`.
        name_tab: Label used in the names of the written files.
        output_dir: If given, the M value matrix of every comparison is written there as
            ``MValue_<comparison>_<name_tab>.csv`` (R writes to ``output/differential``).

    Returns:
        ``{"DifferentialExpression": {comparison: {...}}, "DifferentialMappings": {...}}`` where
        every comparison holds ``Induced_M_Value`` and ``Repressed_M_Value`` (DataFrames of the
        selected probe sets x samples), ``Induced_ProbeID`` and ``Repressed_ProbeID`` (lists), and
        ``Induced_Genes`` and ``Repressed_Genes`` (probe set -> symbols). ``DifferentialMappings``
        maps every probe set selected in any comparison to its symbols.
    """
    samples = _as_samples(order_sample_timeseries)
    logger.info(
        "identify_differentially_expressed_genes: cutoff_induction=%s, cutoff_repression=%s, majority=%s",
        cutoff_induction, cutoff_repression, majority,
    )
    matrices = _m_values(samples, need_log2_scale)
    mappings = probeset_gene_name_mappings
    if mappings is None:
        mappings = map_probeset_names(list(next(iter(samples.values())).index))

    differential: Dict[str, Dict[str, Any]] = {}
    for name, matrix in matrices.items():
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            matrix.to_csv(Path(output_dir) / f"MValue_{name}_{name_tab}.csv")
        induced, repressed = _passing(matrix, cutoff_induction, cutoff_repression, majority)
        induced_ids, repressed_ids = matrix.index[induced].tolist(), matrix.index[repressed].tolist()
        differential[name] = {
            "Induced_M_Value": matrix[induced],
            "Repressed_M_Value": matrix[repressed],
            "Induced_ProbeID": induced_ids,
            "Repressed_ProbeID": repressed_ids,
            "Induced_Genes": {p: mappings.get(p, []) for p in induced_ids},
            "Repressed_Genes": {p: mappings.get(p, []) for p in repressed_ids},
        }
    common = list(dict.fromkeys(
        p for entry in differential.values() for p in entry["Induced_ProbeID"] + entry["Repressed_ProbeID"]
    ))
    return {
        "DifferentialExpression": differential,
        "DifferentialMappings": {p: mappings.get(p, []) for p in common},
    }


# --------------------------------------------------------------------------- gene-level time series

def get_specific_expressed_genes(
    order_sample_timeseries: SampleSeries,
    genelist: Sequence[str] = (),
    annotation: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Probe sets -> gene symbols, keeping the genes present in every sample (R ``getSpecificExpressedGenes``).

    Probe sets without annotation are dropped, rows are renamed to gene symbols (the first probe
    set of a symbol is kept), and only symbols found in every sample remain. `genelist` restricts the
    result to the given genes; probe set ids in it are translated to symbols, symbols are used as given.

    Returns ``KnownDiffExpressedTimeSeriesData`` (sample -> genes x timepoints),
    ``UnKnownDiffExpressedTimeSeriesData`` (always empty: R never fills it),
    ``AllDiffExpressedTimeSeriesData`` and ``ProbesetMapping`` (:class:`AnnotationMappedObject`).
    """
    samples = _as_samples(order_sample_timeseries)
    all_probesets = list(dict.fromkeys(p for f in samples.values() for p in f.index))
    mapping = map_to_gene_name_with_hgu133plus2_db(all_probesets, annotation)
    known: Dict[str, pd.DataFrame] = {}
    if mapping is not None:
        symbol_of = dict(zip(mapping.MappedProbeset["PROBEID"], mapping.MappedProbeset["SYMBOL"]))
        for name, frame in samples.items():
            rows = frame.loc[[p for p in frame.index if p in symbol_of]]
            renamed = rows.rename(index=symbol_of)
            known[name] = renamed[~renamed.index.duplicated(keep="first")]
    common: Optional[List[str]] = None
    for frame in known.values():
        names = list(frame.index)
        common = names if common is None else [n for n in names if n in set(common)]
    if common is not None:
        known = {name: frame.loc[common] for name, frame in known.items()}
    if len(genelist) > 0 and mapping is not None:
        wanted = set(genelist)
        translated = set(mapping.MappedProbeset.loc[mapping.MappedProbeset["PROBEID"].isin(wanted), "SYMBOL"])
        wanted = wanted | translated
        known = {name: frame[frame.index.isin(wanted)] for name, frame in known.items()}
    return {
        "KnownDiffExpressedTimeSeriesData": known,
        "UnKnownDiffExpressedTimeSeriesData": {},
        "AllDiffExpressedTimeSeriesData": dict(known),
        "ProbesetMapping": mapping,
    }


def convert_timeseries_probeset_name_to_gene_name(
    order_sample_timeseries: SampleSeries,
    annotation: Optional[pd.DataFrame] = None,
    mapping: Any = None,
) -> Dict[str, Any]:
    """Renames probe sets to gene symbols (R ``convertTimeseriesProbsetNameToGeneName``).

    Probe sets without a symbol are dropped and probe sets of the same gene are averaged. A probe set
    with several symbols contributes to each of them (R would build a ``c("A", "B")`` string name).
    Symbols come from :func:`map_probeset_names`: `annotation` (a DataFrame with ``PROBEID``,
    ``SYMBOL``, ``ENTREZID`` and ``GENENAME``, deduplicated by Entrez id as in R) or `mapping` (a
    dict, Series or DataFrame of probe set -> symbol) first, then ``DAVID_Gene_List``.

    Returns ``convert_data`` (sample -> genes x timepoints), ``ProbesetMapping``
    (:class:`AnnotationMappedObject` or None), ``mappingsheet`` (probe set -> representative) and
    ``alldifferExpressednames`` (every probe set seen).
    """
    samples = _as_samples(order_sample_timeseries)
    all_names = list(dict.fromkeys(p for f in samples.values() for p in f.index))
    annotated = map_to_gene_name_with_hgu133plus2_db(all_names, annotation) if annotation is not None else None
    symbols = map_probeset_names(all_names, annotated if mapping is None else mapping)

    convert_data: Dict[str, pd.DataFrame] = {}
    for name, frame in samples.items():
        rows, genes = [], []
        for position, probeset in enumerate(frame.index):
            for gene in symbols[probeset]:
                if gene != "N/A":
                    rows.append(position)
                    genes.append(gene)
        expanded = frame.iloc[rows].copy()
        collapsed = expanded.groupby(np.array(genes, dtype=object)).mean()
        collapsed.index.name = None
        convert_data[name] = collapsed
    return {
        "convert_data": convert_data,
        "ProbesetMapping": annotated,
        "mappingsheet": None if annotated is None else annotated.DeDuplicatedMapping,
        "alldifferExpressednames": all_names,
    }


# --------------------------------------------------------------------------- raw data

@dataclass
class AffyBatch:
    """Probe-level raw data (the part of R's ``AffyBatch`` that the pipeline uses)."""

    probeset_names: List[str]
    probe_counts: np.ndarray
    intensities: np.ndarray  # PM cells (all probe sets, consecutive) x arrays
    sample_names: List[str]


def get_related_affy_raw_data(
    cell_directory: PathLike,
    cdf_path: PathLike,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Reads all ``.CEL`` files of a directory (R ``getRelatedAffyRawData``).

    R selects the chip definition by its package name (``cdfname``); here `cdf_path` is the text
    CDF file itself. Returns ``AffyBatchObject`` (:class:`AffyBatch`), ``ExpressionData`` (PM
    intensities, one row per probe, indexed by probe set), ``SampleName`` (file names) and ``Probes``
    (probe set names).
    """
    from .cel_import import read_pm_intensities

    # One listing with a case-insensitive extension: on Windows a "*.CEL" and a "*.cel" glob match the same files.
    files = sorted(
        (path for path in Path(cell_directory).iterdir() if path.is_file() and path.suffix.lower() == ".cel"),
        key=lambda path: (path.suffix != ".CEL", path.name),
    )
    if not files:
        raise ValueError(f"No .CEL files in {cell_directory}")
    names, counts, intensities = read_pm_intensities(files, cdf_path, workers)
    sample_names = [f.name for f in files]
    batch = AffyBatch(names, counts, intensities, sample_names)
    expression = pd.DataFrame(intensities, index=np.repeat(names, counts), columns=sample_names)
    return {"AffyBatchObject": batch, "ExpressionData": expression, "SampleName": sample_names, "Probes": names}


def normalize_timeseries_raw_data(raw_data: Dict[str, Any], type_of_rma: str = "RMA") -> Dict[str, Any]:
    """Normalises raw data to log2 expression (R ``normalizeTimesereisRawData``).

    Only ``"RMA"`` is implemented (quantile normalisation, log2, median polish; no background
    correction). ``"GCRMA"`` and ``"MAS5"`` need the Bioconductor ``gcrma``/``affy`` packages and raise
    ``NotImplementedError``. Returns ``NormalizedObject`` (None), ``NormalizedExpData`` (probe sets x
    arrays), ``SampleName`` and ``FeatureNames``.
    """
    if type_of_rma.upper() != "RMA":
        raise NotImplementedError(f"{type_of_rma} normalisation is not available; use 'RMA'")
    from .cel_import import rma_summarise

    batch: AffyBatch = raw_data["AffyBatchObject"]
    expression = rma_summarise(batch.intensities, batch.probe_counts, "medianpolish")
    normalised = pd.DataFrame(expression, index=batch.probeset_names, columns=batch.sample_names)
    return {
        "NormalizedObject": None,
        "NormalizedExpData": normalised,
        "SampleName": batch.sample_names,
        "FeatureNames": batch.probeset_names,
    }


def convert_affy_raw_data_into_normalized_structure_data(
    files: Union[PathLike, Sequence[PathLike]],
    cdf_path: PathLike,
    use_gcrma: bool = False,
    is_new: bool = True,
    output_dir: Optional[PathLike] = None,
    workers: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """CEL directories -> normalised time series per sample (R ``convertAffyRawDataIntoNormalizedStructureData``).

    For every directory the arrays are read, normalised, grouped into samples by file name and
    ordered by time; the samples of all directories are combined (:func:`dissolve`).

    Args:
        files: One directory or a list of directories with ``.CEL`` files.
        cdf_path: Text CDF file of the array.
        use_gcrma: Not supported (raises ``NotImplementedError`` when True), see
            :func:`normalize_timeseries_raw_data`.
        is_new: Recompute even when a normalised file exists in `output_dir`.
        output_dir: Optional folder for the normalised matrices (``NormalizedEValue_RMA_<dir>.csv``),
            reused on the next call unless `is_new`.
    """
    type_of_rma = "GCRMA" if use_gcrma else "RMA"
    directories = [files] if isinstance(files, (str, Path)) else list(files)
    combined: List[Dict[str, pd.DataFrame]] = []
    for directory in directories:
        logger.info("Processing directory %s", directory)
        cache = None
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            cache = Path(output_dir) / f"NormalizedEValue_{type_of_rma}_{Path(directory).name}.csv"
        if cache is not None and cache.exists() and not is_new:
            normalised = pd.read_csv(cache, index_col=0)
        else:
            raw = get_related_affy_raw_data(directory, cdf_path, workers)
            normalised = normalize_timeseries_raw_data(raw, type_of_rma)["NormalizedExpData"]
            if cache is not None:
                normalised.to_csv(cache)
        combined.append(reorder_sample_timeseries(convert_into_sample_timeseries(normalised)))
    return dissolve(combined)
