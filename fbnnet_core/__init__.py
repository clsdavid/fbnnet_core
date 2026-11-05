"""FBNNet: Fundamental Boolean Network inference from time-series data.

Python port of the R package FBNNet with C++ (pybind11) accelerated mining.
The most common entry points are re-exported here; the compiled extension
modules are private (``fbnnet_core._core`` and friends).

>>> from fbnnet_core import generate_fbm_network
>>> network = generate_fbm_network(timeseries_data, max_k=4)  # doctest: +SKIP
"""
import logging
from importlib.metadata import PackageNotFoundError, version as _version

from .application import binarize_time_series, generate_fbm_network
from .attractor import reconstruct_timeseries, search_for_attractors
from .bioinformatics import (
    convert_affy_raw_data_into_normalized_structure_data,
    convert_into_sample_timeseries,
    convert_timeseries_probeset_name_to_gene_name,
    get_related_affy_raw_data,
    get_specific_expressed_genes,
    identify_differentially_expressed_genes,
    load_hgu133plus2_annotation,
    map_probeset_names,
    map_to_gene_name_with_hgu133plus2_db,
    normalize_timeseries_raw_data,
    reorder_sample_timeseries,
)
from .boolnet import load_network
from .cel_import import cel_to_timeseries
from .cube import construct_fbn_cube
from .datasets import available_datasets, load_dataset
from .geo import download_genome_data_exprs, download_geo_raw_files, download_hgu133plus2_annotation
from .network import merge_cluster_networks, mine_fbn_network

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = _version("fbnnet-core")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "available_datasets",
    "binarize_time_series",
    "cel_to_timeseries",
    "construct_fbn_cube",
    "convert_affy_raw_data_into_normalized_structure_data",
    "convert_into_sample_timeseries",
    "convert_timeseries_probeset_name_to_gene_name",
    "download_genome_data_exprs",
    "download_geo_raw_files",
    "download_hgu133plus2_annotation",
    "generate_fbm_network",
    "get_related_affy_raw_data",
    "get_specific_expressed_genes",
    "identify_differentially_expressed_genes",
    "load_dataset",
    "load_hgu133plus2_annotation",
    "load_network",
    "map_probeset_names",
    "map_to_gene_name_with_hgu133plus2_db",
    "merge_cluster_networks",
    "mine_fbn_network",
    "normalize_timeseries_raw_data",
    "reorder_sample_timeseries",
    "reconstruct_timeseries",
    "search_for_attractors",
]
