"""Example data sets shipped with the package (converted from the R package FBNNet).

>>> from fbnnet_core.datasets import load_dataset
>>> network = load_dataset("FBM_Leukeamia_Networks")   # a Fundamental Boolean Network
"""
import gzip
import json
from importlib import resources
from typing import Any, Dict, List

import pandas as pd

from .fbn_types import FundamentalBooleanNetwork

_DESCRIPTIONS = {
    "BoolNet_CellCycle_Network": "Mammalian cell-cycle Boolean network (BoolNet format) from Fauré et al. (2006).",
    "ExampleNetwork": "Five-gene Boolean network used throughout the paper and the vignette (BoolNet format).",
    "ExampleTimeseriesData": "Time series (32 samples of 5 genes) generated from a fundamental Boolean model.",
    "FBNExampleNetworks": "Fundamental Boolean network of the five-gene example.",
    "FBNcellcycleNetwork": "Fundamental Boolean network of the mammalian cell cycle (Chen et al., 2018).",
    "FBM_Leukeamia_Networks": "Leukaemia network, timestep 1 only (Chen et al., 2022).",
    "TFBM_Leukeamia_Networks": "Leukaemia network with temporal rules up to timestep 2 (Chen et al., 2022).",
    "Leukeamia_Networks": "Leukaemia network with temporal rules up to timestep 2 (Chen et al., 2022).",
    "Common_Genes_Leukeamia": "The 285 genes common to the leukaemia study (Chen et al., 2022).",
    "Common_Genes_Clusters": "DAVID functional annotation clusters of the common genes.",
    "Leukeamia_Timeseries": "Binarised time series (26 patients x 285 genes) of the leukaemia study.",
    "DAVID_Gene_List": "DAVID annotation mapping probeset ids to gene symbols and names.",
    "yeastTimeSeries": "Yeast cell-cycle expression of four genes over 14 time points.",
}
_ALIASES = {"fbm_leukeamia_network": "FBM_Leukeamia_Networks", "DAVID_gene_list": "DAVID_Gene_List"}


def available_datasets() -> Dict[str, str]:
    """Return ``{dataset name: description}`` for every bundled data set."""
    return dict(_DESCRIPTIONS)


def _frame(spec: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(spec["data"], columns=spec["columns"])


def _matrix(spec: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(spec["values"], index=spec["genes"], columns=spec["columns"])


def load_dataset(name: str):
    """Load one of the bundled data sets.

    Returns, depending on the data set: a ``FundamentalBooleanNetwork`` (``*Networks``, ``FBN*``), a
    BoolNet-style network ``dict`` (``ExampleNetwork``, ``BoolNet_CellCycle_Network``), a list of
    genes x time points ``DataFrame`` (``*Timeseries*``), a single ``DataFrame`` (``yeastTimeSeries``,
    ``DAVID_Gene_List``), a list of gene names, or a dict of ``DataFrame`` (``Common_Genes_Clusters``).
    """
    name = _ALIASES.get(name, name)
    if name not in _DESCRIPTIONS:
        raise ValueError(f"Unknown data set '{name}'. Available: {', '.join(sorted(_DESCRIPTIONS))}")

    path = resources.files(__package__).joinpath(f"datasets/{name}.json.gz")
    with path.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    kind, content = payload["kind"], payload["content"]

    if kind == "fbn_network":
        content["class"] = "FundamentalBooleanNetwork"
        return FundamentalBooleanNetwork(content)
    if kind == "boolnet_network":
        return content
    if kind == "timeseries_list":
        series: List[pd.DataFrame] = []
        for sample, spec in zip(content["names"], content["matrices"]):
            frame = _matrix(spec)
            frame.attrs["sample"] = sample
            series.append(frame)
        return series
    if kind == "timeseries_matrix":
        return _matrix(content)
    if kind == "gene_list":
        return content
    if kind == "table":
        return _frame(content)
    if kind == "table_dict":
        return {key: _frame(spec) for key, spec in content.items()}
    raise ValueError(f"Unsupported data set kind '{kind}'")
