"""Convert the R package's datasets (data/*.rda) into the portable JSON files shipped in fbnnet_core/datasets/.

Run from the repository root (needs `rdata`): python tools/convert_r_datasets.py
"""
import gzip
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import rdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fbnnet_core", "datasets")

# file stem -> (object name in the .rda, kind)
DATASETS = {
    "BoolNet_CellCycle_Network": ("BoolNet_CellCycle_Network", "boolnet_network"),
    "ExampleNetwork": ("ExampleNetwork", "boolnet_network"),
    "ExampleTimeseriesData": ("ExampleTimeseriesData", "timeseries_list"),
    "FBNExampleNetworks": ("FBNExampleNetworks", "fbn_network"),
    "FBNcellcycleNetwork": ("FBNcellcycleNetwork", "fbn_network"),
    "FBM_Leukeamia_Networks": ("FBM_Leukeamia_Networks", "fbn_network"),
    "TFBM_Leukeamia_Networks": ("TFBM_Leukeamia_Networks", "fbn_network"),
    "Leukeamia_Networks": ("Leukeamia_Networks", "fbn_network"),
    "Common_Genes_Leukeamia": ("Common_Genes_Leukeamia", "gene_list"),
    "Common_Genes_Clusters": ("Common_Genes_Clusters", "table_dict"),
    "Leukeamia_Timeseries": ("Leukeamia_Timeseries", "timeseries_list"),
    "DAVID_Gene_List": ("DAVID_Gene_List", "table"),
    "yeastTimeSeries": ("yeastTimeSeries", "timeseries_matrix"),
}
# the .rda file name where it differs from the dataset name
FILES = {"DAVID_Gene_List": "DAVID_gene_list"}


def scalar(value):
    value = value.tolist() if hasattr(value, "tolist") else value
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


_NUMERIC_RULE_FIELDS = ("error", "type", "probability", "support", "timestep")


def rule_value(key, value):
    if key == "input":
        return [int(i) for i in np.ravel(value)]
    value = scalar(value)
    if key in _NUMERIC_RULE_FIELDS:
        return float(value)  # R stores some of these as character strings
    return value


def fbn_network(obj):
    genes = [str(g) for g in obj["genes"]]
    interactions = {}
    for gene in genes:
        rules = obj["interactions"][gene]
        rules = rules if hasattr(rules, "items") else {}  # a gene without rules is an empty list
        interactions[gene] = {
            str(name): {k: rule_value(k, v) for k, v in rule.items()}
            for name, rule in rules.items()
        }
    return {
        "genes": genes,
        "interactions": interactions,
        "fixed": dict(zip(genes, [float(x) for x in np.ravel(obj["fixed"])])),
        "timedecay": dict(zip(genes, [float(x) for x in np.ravel(obj["timedecay"])])),
    }


def boolnet_network(obj):
    """Same structure as fbnnet_core.boolnet.load_network, built from the stored expressions."""
    from fbnnet_core.boolnet import load_network

    genes = [str(g) for g in obj["genes"]]
    lines = ["targets, factors"] + [f"{g}, {scalar(obj['interactions'][g]['expression'])}" for g in genes]
    with tempfile.NamedTemporaryFile("w", suffix=".bn", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
    try:
        network = load_network(handle.name)
    finally:
        os.unlink(handle.name)
    return {"genes": network["genes"], "interactions": dict(network["interactions"]),
            "expressions": network["expressions"]}


def matrix(array):
    genes = [str(g) for g in array.coords["dim_0"].values]
    if "dim_1" in array.coords:
        columns = [str(c) for c in array.coords["dim_1"].values]
    else:
        columns = [str(i + 1) for i in range(array.shape[1])]
    return {"genes": genes, "columns": columns, "values": np.asarray(array.values, dtype=float).tolist()}


def table(frame):
    frame = frame.reset_index(drop=True)
    return {"columns": [str(c) for c in frame.columns], "data": json.loads(frame.to_json(orient="values"))}


def convert(kind, obj):
    if kind == "fbn_network":
        return fbn_network(obj)
    if kind == "boolnet_network":
        return boolnet_network(obj)
    if kind == "timeseries_list":
        return {"names": [str(k) for k in obj.keys()], "matrices": [matrix(v) for v in obj.values()]}
    if kind == "timeseries_matrix":
        return matrix(obj)
    if kind == "gene_list":
        return [str(g) for g in obj]
    if kind == "table":
        return table(obj)
    if kind == "table_dict":
        return {str(k): table(v) for k, v in obj.items()}
    raise ValueError(kind)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (obj_name, kind) in DATASETS.items():
        path = os.path.join(ROOT, "data", f"{FILES.get(name, name)}.rda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = rdata.conversion.convert(rdata.parser.parse_file(path))[obj_name]
        payload = {"name": name, "kind": kind, "content": convert(kind, obj)}
        target = os.path.join(OUT, f"{name}.json.gz")
        with gzip.open(target, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, separators=(",", ":"))
        print(f"{name:28s} {kind:18s} {os.path.getsize(target):>9,d} bytes")


if __name__ == "__main__":
    main()
