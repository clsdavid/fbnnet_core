# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 Leshi Chen
"""Convert raw Affymetrix CEL files into genome-wide Boolean time series.

The result is the input of :func:`fbnnet_core.construct_fbn_cube` /
:func:`fbnnet_core.generate_fbm_network`: a list with one DataFrame per
sample (patient, replicate, ...), genes on the rows and timepoints on the
columns.

No Bioconductor is needed, only NumPy and pandas. The pipeline follows the R
package (``R/FBNBioinformatics.R``, ``R/sample_FBN.R``, ``R/FBNLeukaemiaStudy.R``):

1. read the chip definition (CDF) to find the perfect-match (PM) cells of
   every probe set, and read every CEL file (``getRelatedAffyRawData``);
2. quantile-normalise across arrays, take log2 and summarise every probe set
   with median polish (``normalizeTimesereisRawData``, i.e. RMA **without** its
   background-correction step, like ``affy::rma(background = FALSE)``);
3. group arrays into samples by file name and order them by time
   (``convertIntoSampleTimeSeries`` / ``reorderSampleTimeSeries``);
4. optionally keep the differentially expressed probe sets: a log2 change of at
   least ``cutoff`` between two timepoints in at least ``majority`` samples
   (``identifyDifferentiallyExpressedGenes``);
5. optionally map probe sets to gene symbols, dropping unmapped probe sets and
   averaging probe sets of the same gene (``convertTimeseriesProbsetNameToGeneName``);
6. optionally keep the N most variable genes, then binarise every gene with
   :func:`fbnnet_core.binarize_time_series` (``BoolNet::binarizeTimeSeries``).

Differences from R: no RMA background correction or GCRMA, and the probe set to
gene mapping uses the bundled ``DAVID_Gene_List`` (R tries ``hgu133plus2.db``
first); pass ``mapping`` (for example the path of ``hgu133plus2.sqlite``) to use another annotation. Supported inputs are text
CDF files (``[CDF]`` / GC3.0) and text (version 3) CEL files.

Example (13 patients measured at three timepoints, file names such as
``B-ALL-13-0h.CEL``)::

    from fbnnet_core.cel_import import cel_to_timeseries
    samples = cel_to_timeseries(
        "GSE2677_RAW", "HG-U133_Plus_2.cdf",
        differential=dict(cutoff_induction=0.7, cutoff_repression=0.7, majority=7),
        map_genes=True,
    )

Command line::

    python -m fbnnet_core.cel_import GSE2677_RAW HG-U133_Plus_2.cdf out.npz --differential 0.7 0.7 7 --map-genes

On platforms that start workers with ``spawn`` (Windows, macOS) call these
functions from code guarded by ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .bioinformatics import convert_timeseries_probeset_name_to_gene_name, differentially_expressed_probesets

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

DEFAULT_SAMPLE_PATTERN = r"^(?P<sample>.+)[-_](?P<time>\d+(?:\.\d+)?)h\.CEL$"
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def read_cdf_pm_cells(cdf_path: PathLike) -> Tuple[List[str], List[np.ndarray]]:
    """Probe set names and, for each, the CEL cell indices of its PM probes.

    A cell is a perfect match when its probe base is the complement of its
    target base (a mismatch probe repeats the target base in the middle).
    Cell indices are row-major positions ``y * columns + x``.
    """
    names: List[str] = []
    cells: List[List[int]] = []
    in_block = False
    with open(cdf_path, "r", errors="replace") as handle:
        if not handle.readline().startswith("[CDF]"):
            raise ValueError(f"{cdf_path} is not a text CDF file (binary CDF is not supported)")
        for line in handle:
            if line.startswith("["):
                in_block = line.startswith("[Unit") and "_Block" in line
                if in_block:
                    names.append("")
                    cells.append([])
            elif in_block:
                if line.startswith("Name="):
                    names[-1] = line[5:].strip()
                elif line.startswith("Cell"):
                    # Cell<n>=X Y PROBE FEAT QUAL EXPOS POS CBASE PBASE TBASE ATOM INDEX ...
                    fields = line.split("\t")
                    if _COMPLEMENT.get(fields[9]) == fields[8]:
                        cells[-1].append(int(fields[11]))
    return names, [np.asarray(c, dtype=np.int64) for c in cells]


def read_cel_intensities(cel_path: PathLike) -> np.ndarray:
    """Cell intensities of a text (version 3) CEL file in row-major cell order."""
    columns = n_cells = skip = None
    with open(cel_path, "rb") as handle:
        if handle.read(1) == b"\x40":
            raise ValueError(f"{cel_path} is a binary CEL file; only text (version 3) CEL is supported")
    with open(cel_path, "r", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.startswith("Cols="):
                columns = int(line.split("=")[1])
            elif line.startswith("NumberCells="):
                n_cells = int(line.split("=")[1])
            elif line.startswith("CellHeader="):
                skip = number + 1
                break
    if columns is None or n_cells is None or skip is None:
        raise ValueError(f"{cel_path} is not a valid text CEL file")
    table = pd.read_csv(
        cel_path, sep="\t", skiprows=skip, nrows=n_cells, header=None,
        usecols=[0, 1, 2], names=["x", "y", "mean"], dtype=np.float64,
    )
    values = np.empty(n_cells, dtype=np.float32)
    values[table["y"].to_numpy(np.int64) * columns + table["x"].to_numpy(np.int64)] = table["mean"].to_numpy()
    return values


def quantile_normalise(matrix: np.ndarray) -> np.ndarray:
    """Quantile normalisation of the columns of `matrix`.

    Tied values in a column all receive the mean of the reference quantiles over their
    rank range (as ``preprocessCore::normalize.quantiles`` does), so ties are order-free.
    """
    ordered = np.sort(matrix, axis=0)
    reference = ordered.mean(axis=1)
    result = np.empty(matrix.shape, dtype=np.float64)
    for column in range(matrix.shape[1]):
        sorted_values = ordered[:, column]
        group = np.concatenate(([0], np.cumsum(sorted_values[1:] != sorted_values[:-1])))
        group_mean = np.bincount(group, weights=reference) / np.bincount(group)
        result[np.argsort(matrix[:, column], kind="stable"), column] = group_mean[group]
    return result


def _read_pm(args: Tuple[str, np.ndarray]) -> np.ndarray:
    path, pm_index = args
    return read_cel_intensities(path)[pm_index]


def median_polish_summary(log_values: np.ndarray, counts: np.ndarray, max_iter: int = 10, eps: float = 0.01) -> np.ndarray:
    """RMA summarisation: Tukey median polish of every probe set (probes x arrays).

    `log_values` stacks the probes of all probe sets (``counts[i]`` consecutive rows
    for probe set i). Returns one value per probe set and array (overall + array effect).
    Probe sets with the same number of probes are polished together.
    """
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    result = np.empty((len(counts), log_values.shape[1]))
    for n_probes in np.unique(counts):
        members = np.flatnonzero(counts == n_probes)
        for start in range(0, len(members), 4096):
            chunk = members[start:start + 4096]
            rows = offsets[chunk][:, None] + np.arange(n_probes)[None, :]
            z = log_values[rows]  # (sets, probes, arrays)
            row_effect = np.zeros(z.shape[:2])
            col_effect = np.zeros((z.shape[0], z.shape[2]))
            overall = np.zeros(z.shape[0])
            previous = np.abs(z).sum(axis=(1, 2))
            for _ in range(max_iter):
                delta = np.median(z, axis=2)
                z -= delta[:, :, None]
                row_effect += delta
                shift = np.median(col_effect, axis=1)
                col_effect -= shift[:, None]
                overall += shift
                delta = np.median(z, axis=1)
                z -= delta[:, None, :]
                col_effect += delta
                shift = np.median(row_effect, axis=1)
                row_effect -= shift[:, None]
                overall += shift
                current = np.abs(z).sum(axis=(1, 2))
                done = np.all(np.abs(current - previous) <= eps * np.maximum(current, 1e-12))
                previous = current
                if done:
                    break
            result[chunk] = overall[:, None] + col_effect
    return result


def read_pm_intensities(
    cel_files: Sequence[PathLike],
    cdf_path: PathLike,
    workers: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """PM intensities of CEL files: (probe set names, probes per probe set, probes x arrays)."""
    names, cells = read_cdf_pm_cells(cdf_path)
    counts = np.array([len(c) for c in cells])
    keep = counts > 0
    names = [n for n, k in zip(names, keep) if k]
    counts = counts[keep]
    pm_index = np.concatenate([c for c, k in zip(cells, keep) if k])
    logger.info("CDF: %d probe sets, %d PM cells", len(names), len(pm_index))

    n_workers = max(1, min(workers or os.cpu_count() or 1, len(cel_files)))
    jobs = [(str(f), pm_index) for f in cel_files]
    if n_workers == 1:
        columns = [_read_pm(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            columns = list(pool.map(_read_pm, jobs))
    return names, counts, np.stack(columns, axis=1)


def rma_summarise(intensities: np.ndarray, counts: np.ndarray, summary: str = "medianpolish") -> np.ndarray:
    """Quantile normalisation, log2 and per-probe-set summary of PM intensities (probe sets x arrays)."""
    if summary not in ("medianpolish", "mean"):
        raise ValueError("summary must be 'medianpolish' or 'mean'")
    log_values = np.log2(np.maximum(quantile_normalise(intensities), 1.0))
    if summary == "mean":
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
        return np.add.reduceat(log_values, offsets, axis=0) / counts[:, None]
    return median_polish_summary(log_values, counts)


def summarise_expression(
    cel_files: Sequence[PathLike],
    cdf_path: PathLike,
    workers: Optional[int] = None,
    summary: str = "medianpolish",
) -> pd.DataFrame:
    """log2 expression (probe sets x arrays) from CEL files; columns are the file names.

    `summary` is ``"medianpolish"`` (RMA, default) or ``"mean"`` (mean of the log2 PM probes).
    """
    if summary not in ("medianpolish", "mean"):
        raise ValueError("summary must be 'medianpolish' or 'mean'")
    started = time.perf_counter()
    names, counts, intensities = read_pm_intensities(cel_files, cdf_path, workers)
    expression = rma_summarise(intensities, counts, summary)
    logger.info("Summarised %d arrays in %.1fs", len(cel_files), time.perf_counter() - started)
    return pd.DataFrame(expression, index=names, columns=[Path(f).name for f in cel_files])


def group_cel_files(
    cel_dir: PathLike,
    sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
    n_timepoints: Optional[int] = None,
) -> Dict[str, List[Path]]:
    """Group CEL files into samples, each ordered by the time in its file name.

    `sample_pattern` is a regular expression with the named groups ``sample`` and ``time``.
    Samples that do not have `n_timepoints` arrays are dropped (default: the most
    common count), so every returned sample has the same number of timepoints.
    """
    pattern = re.compile(sample_pattern, re.IGNORECASE)
    grouped: Dict[str, List[Tuple[float, Path]]] = {}
    for path in sorted(Path(cel_dir).iterdir()):
        match = pattern.match(path.name)
        if match:
            grouped.setdefault(match["sample"], []).append((float(match["time"]), path))
    if not grouped:
        raise ValueError(f"No CEL files in {cel_dir} match the sample pattern {sample_pattern!r}")
    if n_timepoints is None:
        sizes = [len(v) for v in grouped.values()]
        n_timepoints = max(set(sizes), key=sizes.count)
    dropped = sorted(s for s, v in grouped.items() if len(v) != n_timepoints)
    if dropped:
        logger.warning("Dropping samples without %d timepoints: %s", n_timepoints, ", ".join(dropped))
    return {s: [p for _, p in sorted(v)] for s, v in sorted(grouped.items()) if len(v) == n_timepoints}


def expression_to_timeseries(
    expression: pd.DataFrame,
    samples: Dict[str, List[Path]],
    n_genes: int = 0,
    binarise: bool = True,
    exclude_prefixes: Sequence[str] = ("AFFX",),
    differential: Optional[dict] = None,
    map_genes: bool = False,
    mapping=None,
) -> List[pd.DataFrame]:
    """Split a probe-set expression matrix into one genes x timepoints DataFrame per sample.

    See :func:`cel_to_timeseries` for the options; the steps run in the order
    differential selection, gene mapping, top-variance selection, binarisation.
    """
    if exclude_prefixes:
        expression = expression[~expression.index.str.startswith(tuple(exclude_prefixes))]
    n_timepoints = len(next(iter(samples.values())))
    labels = [f"t{i + 1}" for i in range(n_timepoints)]
    series = [
        pd.DataFrame(expression[[p.name for p in paths]].to_numpy(), index=expression.index, columns=labels)
        for paths in samples.values()
    ]
    if differential is not None:
        options = dict(differential)
        chosen = options.pop("samples", None)
        tested = [f for name, f in zip(samples, series) if chosen is None or name in chosen]
        keep = differentially_expressed_probesets(tested, **options)
        series = [frame.loc[keep] for frame in series]
    if map_genes:
        series = list(convert_timeseries_probeset_name_to_gene_name(series, mapping=mapping)["convert_data"].values())
    if n_genes and n_genes < len(series[0]):
        stacked = np.stack([f.to_numpy(dtype=float) for f in series], axis=0)
        variance = stacked.transpose(1, 0, 2).reshape(stacked.shape[1], -1).var(axis=1)
        top = np.sort(np.argsort(variance)[::-1][:n_genes])
        series = [frame.iloc[top] for frame in series]
    if not binarise:
        return series
    from .application import binarize_time_series

    return binarize_time_series(series)


def cel_to_timeseries(
    cel_dir: PathLike,
    cdf_path: PathLike,
    *,
    sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
    n_timepoints: Optional[int] = None,
    differential: Optional[dict] = None,
    map_genes: bool = False,
    mapping=None,
    n_genes: int = 0,
    binarise: bool = True,
    summary: str = "medianpolish",
    workers: Optional[int] = None,
    cache_path: Optional[PathLike] = None,
    exclude_prefixes: Sequence[str] = ("AFFX",),
) -> List[pd.DataFrame]:
    """Genome-wide time series (one DataFrame per sample) from a directory of CEL files.

    Args:
        cel_dir: Directory with the ``.CEL`` files.
        cdf_path: Text CDF of the array (e.g. ``HG-U133_Plus_2.cdf``).
        sample_pattern: Regex with named groups ``sample`` and ``time`` applied to file names.
        n_timepoints: Required arrays per sample (default: the most common count).
        differential: Keep only differentially expressed probe sets, with the keys
            ``cutoff_induction`` (0.7), ``cutoff_repression`` (0.7), ``majority`` (7) and
            optionally ``samples`` (names of the samples to test; default all), as in the R
            leukaemia study. Default None: keep every probe set.
        map_genes: Rename probe sets to gene symbols (unmapped dropped, duplicates averaged).
        mapping: Annotation for `map_genes`: path of ``hgu133plus2.sqlite``, a DataFrame or a dict of
            probe set -> symbol (default: the bundled DAVID list).
        n_genes: Keep the N most variable genes (0 = all).
        binarise: Return 0/1 values (2-means per gene) instead of log2 expression.
        summary: ``"medianpolish"`` (RMA, default) or ``"mean"``.
        workers: CEL reader processes (default: all cores).
        cache_path: Optional ``.pkl`` cache of the summarised expression matrix; reused
            when it covers the same files, so changing the options is fast.
        exclude_prefixes: Probe set name prefixes to drop (Affymetrix controls by default).
    """
    samples = group_cel_files(cel_dir, sample_pattern, n_timepoints)
    files = [p for paths in samples.values() for p in paths]
    names = [p.name for p in files]

    expression = None
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_pickle(cache_path)
        if list(cached.columns) == names and cached.attrs.get("summary") == summary:
            expression = cached
    if expression is None:
        expression = summarise_expression(files, cdf_path, workers, summary)
        expression.attrs["summary"] = summary
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            expression.to_pickle(cache_path)
    return expression_to_timeseries(
        expression, samples, n_genes, binarise, exclude_prefixes, differential, map_genes, mapping
    )


def save_timeseries(samples: List[pd.DataFrame], path: PathLike) -> None:
    """Writes time series to a compressed ``.npz`` (values, genes, timepoints, sample names)."""
    np.savez_compressed(
        path,
        data=np.stack([s.to_numpy() for s in samples]),
        genes=np.array(samples[0].index, dtype=str),
        timepoints=np.array(samples[0].columns, dtype=str),
    )


def load_timeseries(path: PathLike) -> List[pd.DataFrame]:
    """Reads time series written by :func:`save_timeseries`."""
    stored = np.load(path, allow_pickle=False)
    genes, timepoints = stored["genes"].tolist(), stored["timepoints"].tolist()
    return [pd.DataFrame(matrix, index=genes, columns=timepoints) for matrix in stored["data"]]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m fbnnet_core.cel_import",
        description="Convert Affymetrix CEL files into genome-wide Boolean time series (.npz).",
    )
    parser.add_argument("cel_dir")
    parser.add_argument("cdf")
    parser.add_argument("out", help="output .npz (read it back with load_timeseries)")
    parser.add_argument("--n-genes", type=int, default=0, help="keep the N most variable genes (0 = all)")
    parser.add_argument("--differential", type=float, nargs=3, metavar=("INDUCTION", "REPRESSION", "MAJORITY"),
                        help="keep differentially expressed probe sets, e.g. 0.7 0.7 7 (as in the R leukaemia study)")
    parser.add_argument("--map-genes", action="store_true", help="map probe sets to gene symbols (bundled DAVID list)")
    parser.add_argument("--summary", choices=["medianpolish", "mean"], default="medianpolish")
    parser.add_argument("--raw", action="store_true", help="write log2 expression instead of 0/1 values")
    parser.add_argument("--pattern", default=DEFAULT_SAMPLE_PATTERN, help="file-name regex with groups 'sample' and 'time'")
    parser.add_argument("--timepoints", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", default=None, help="cache file for the summarised expression matrix")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    samples = cel_to_timeseries(
        args.cel_dir, args.cdf, sample_pattern=args.pattern, n_timepoints=args.timepoints,
        n_genes=args.n_genes, binarise=not args.raw, workers=args.workers, cache_path=args.cache,
        summary=args.summary, map_genes=args.map_genes,
        differential=None if args.differential is None else dict(
            cutoff_induction=args.differential[0], cutoff_repression=args.differential[1],
            majority=int(args.differential[2])),
    )
    save_timeseries(samples, args.out)
    print(f"{args.out}: {len(samples)} samples x {samples[0].shape[0]} genes x {samples[0].shape[1]} timepoints")


if __name__ == "__main__":
    main()
