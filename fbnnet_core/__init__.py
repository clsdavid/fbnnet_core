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
from .boolnet import load_network
from .cube import construct_fbn_cube
from .datasets import available_datasets, load_dataset
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
    "construct_fbn_cube",
    "generate_fbm_network",
    "load_dataset",
    "load_network",
    "merge_cluster_networks",
    "mine_fbn_network",
    "reconstruct_timeseries",
    "search_for_attractors",
]
