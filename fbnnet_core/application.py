"""
Top-level pipeline entry point for FBNNet.

Python port of R's `application_FBN.R` (`generateFBMNetwork`), the single
convenience function that chains together (optional) discretisation of raw
time-series data, `construct_fbn_cube`, and `mine_fbn_network`.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .cube import construct_fbn_cube
from .network import mine_fbn_network
from .general_utils import check_numeric, check_probability_type_data

logger = logging.getLogger(__name__)

_VALID_METHODS = ("kmeans", "edgeDetector")
_VALID_EDGES = ("firstEdge", "maxEdge")


def _binarize_series(values: np.ndarray) -> np.ndarray:
    """
    Binarize a single 1-D numeric series into {0, 1} using a simple
    1-dimensional 2-means clustering (lightweight stand-in for
    ``BoolNet::binarizeTimeSeries(method = "kmeans")`` that avoids adding a
    hard dependency on scikit-learn).
    """
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()
    if low == high:
        # Constant series: nothing to discriminate, treat as all "off".
        return np.zeros_like(values, dtype=int)

    centroid_low, centroid_high = low, high
    for _ in range(100):
        dist_low = np.abs(values - centroid_low)
        dist_high = np.abs(values - centroid_high)
        assigned_high = dist_high < dist_low

        new_low = values[~assigned_high].mean() if np.any(~assigned_high) else centroid_low
        new_high = values[assigned_high].mean() if np.any(assigned_high) else centroid_high

        if new_low == centroid_low and new_high == centroid_high:
            break
        centroid_low, centroid_high = new_low, new_high

    return assigned_high.astype(int)


def _edge_detector_series(values: np.ndarray, scaling: float = 1.0, edge: str = "firstEdge") -> np.ndarray:
    """Port of BoolNet's `edgeDetector`: threshold the series at an edge of its sorted values."""
    values = np.asarray(values, dtype=float)
    if edge not in _VALID_EDGES:
        raise ValueError(f"'edge' must be one of {_VALID_EDGES}")

    sorted_values = np.sort(values)
    distance = np.diff(sorted_values)
    if edge == "firstEdge":
        threshold = scaling * (sorted_values[-1] - sorted_values[0]) / (len(values) - 1)
        above = np.flatnonzero(distance > threshold)
        index = int(above[0]) if above.size else None
    else:
        index = int(np.argmax(distance))

    if index is None:
        return np.zeros_like(values, dtype=int)
    return (values >= sorted_values[index + 1]).astype(int)


def binarize_time_series(
    timeseries_data: List[pd.DataFrame],
    method: str = "kmeans",
    edge: str = "firstEdge",
    scaling: float = 1.0,
) -> List[pd.DataFrame]:
    """
    Discretise raw numeric time-series data into boolean (0/1) values, one
    gene (row) at a time.

    Args:
        timeseries_data: A list of DataFrames (genes as rows, timepoints as
            columns).
        method: "kmeans" (per matrix) or "edgeDetector" (BoolNet's edge
            detector; like BoolNet it thresholds each gene over all matrices
            concatenated).
        edge: For "edgeDetector", "firstEdge" or "maxEdge".
        scaling: For "edgeDetector" with "firstEdge", scales the edge size.

    Returns:
        A new list of DataFrames with the same shape/index/columns, but with
        binary values.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unsupported discretisation method '{method}', only {_VALID_METHODS} are supported")

    if method == "edgeDetector":
        genes = timeseries_data[0].index
        widths = [matrix.shape[1] for matrix in timeseries_data]
        collated = np.hstack([matrix.loc[genes].to_numpy(dtype=float) for matrix in timeseries_data])
        bins = np.vstack([_edge_detector_series(row, scaling, edge) for row in collated])
        splits = np.cumsum(widths)[:-1]
        return [
            pd.DataFrame(part, index=genes, columns=matrix.columns)
            for part, matrix in zip(np.hsplit(bins, splits), timeseries_data)
        ]

    binarized = []
    for matrix in timeseries_data:
        result = matrix.copy()
        for gene in result.index:
            result.loc[gene] = _binarize_series(result.loc[gene].to_numpy())
        binarized.append(result.astype(int))
    return binarized


def is_boolean_type_timeseries_data(timeseries_data: List[pd.DataFrame]) -> bool:
    """Check whether every value in every matrix is already 0/1."""
    for matrix in timeseries_data:
        unique_values = np.unique(matrix.to_numpy())
        if not np.all(np.isin(unique_values, [0, 1])):
            return False
    return True


def generate_fbm_network(
    timeseries_data: Union[pd.DataFrame, List[pd.DataFrame]],
    method: str = "kmeans",
    max_k: int = 4,
    use_parallel: bool = False,
    max_deep_temporal: int = 1,
    threshold_confidence: float = 1,
    threshold_error: float = 0,
    threshold_support: float = 1e-5,
    max_fbn_rules: int = 5,
    network_only: bool = True,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    chunksize: Optional[int] = None,
    backend: str = "recursive",
) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Main entry point of the pipeline: mine a Fundamental Boolean Network
    directly from (optionally non-boolean) time-series data.

    Python port of R's ``generateFBMNetwork``.

    Args:
        timeseries_data: A single DataFrame or a list of DataFrames, each with
            genes as rows and timepoints as columns.
        method: Discretisation method used if the data is not already boolean
            ("kmeans" or "edgeDetector"; see ``binarize_time_series``).
        max_k: The maximum depth the Orchard Cube can mine into.
        use_parallel: If True, run the network inference algorithm in
            parallel.
        max_deep_temporal: Maximum temporal space for the Temporal FBN model.
        threshold_confidence: Confidence threshold (0-1) to filter FBN rules.
        threshold_error: Error-rate threshold (0-1) to filter FBN rules.
        threshold_support: Support threshold (0-1) to filter FBN rules.
        max_fbn_rules: Maximum rules per type (Activator/Inhibitor) per gene.
        network_only: If True (default), only return the mined network.
            Otherwise, also return the Orchard cube.
        verbose: If True, emit INFO-level log messages describing progress.
        max_workers: Maximum number of worker processes used when
            use_parallel is True (default: cpu_count - 1).
        chunksize: Tasks-per-worker batch size passed to the underlying
            pool.map when use_parallel is True (default: pool.map's own
            heuristic). Tune this upward for very large gene counts to
            reduce IPC/dispatch overhead.
        backend: "recursive" (default, C++ engine) or "tensor" (NumPy node-level
            batching, identical results, faster on wide gene sets).

    Returns:
        The mined FBN network dict, or ``{"cube": ..., "network": ...}`` if
        ``network_only`` is False.
    """
    if isinstance(timeseries_data, pd.DataFrame):
        timeseries_data = [timeseries_data]

    if not isinstance(timeseries_data, list) or not timeseries_data:
        raise ValueError("timeseries_data must be a non-empty DataFrame or list of DataFrames")
    if not all(isinstance(item, pd.DataFrame) for item in timeseries_data):
        raise ValueError("Every element of timeseries_data must be a pandas DataFrame")

    check_probability_type_data(threshold_confidence)
    check_probability_type_data(threshold_error)
    check_probability_type_data(threshold_support)
    check_numeric(max_fbn_rules)

    log = logger.info if verbose else logger.debug
    log(
        "Enter generate_fbm_network zone: method=%s, max_k=%s, use_parallel=%s, "
        "max_deep_temporal=%s, threshold_confidence=%s, threshold_error=%s, "
        "threshold_support=%s, max_fbn_rules=%s, network_only=%s",
        method, max_k, use_parallel, max_deep_temporal, threshold_confidence,
        threshold_error, threshold_support, max_fbn_rules, network_only,
    )

    if not is_boolean_type_timeseries_data(timeseries_data):
        timeseries_data = binarize_time_series(timeseries_data, method=method)

    genes = list(timeseries_data[0].index)

    log("Run generate_fbm_network with a single cube")
    cube = construct_fbn_cube(
        target_genes=genes,
        conditional_genes=genes,
        timeseries_cube=timeseries_data,
        max_k=max_k,
        temporal=max_deep_temporal,
        use_parallel=use_parallel,
        max_workers=max_workers,
        chunksize=chunksize,
        backend=backend,
    )
    network = mine_fbn_network(
        cube,
        genes=genes,
        use_parallel=use_parallel,
        threshold_confidence=threshold_confidence,
        threshold_error=threshold_error,
        threshold_support=threshold_support,
        max_fbn_rules=max_fbn_rules,
    )

    log("Leave generate_fbm_network zone")
    if network_only:
        return network
    return {"cube": cube, "network": network}
