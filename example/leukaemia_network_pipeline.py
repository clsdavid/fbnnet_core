"""
Reproducible pipeline: mine a Fundamental Boolean Network from the leukaemia time series
(`data/Leukeamia_Timeseries.rda`) with the Python port, record every parameter, and emit an R
script that runs the *same* experiment with the R package FBNNet so the outputs can be diffed.

Run from the repository root:

    python example/leukaemia_network_pipeline.py --scenario non_temporal   # compare with FBM_Leukeamia_Networks.rda
    python example/leukaemia_network_pipeline.py --scenario temporal       # compare with Leukeamia_Networks.rda
    python example/leukaemia_network_pipeline.py --max-k 3 --temporal 1 --out-dir /tmp/leuk   # custom

Outputs (in --out-dir, default example/leukaemia_output/<scenario>):
    params.json            every parameter + data fingerprint + environment versions
    python_rules.csv       one row per mined rule (target,type,expression,timestep,support,error,probability)
    python_network.txt     the network as printed by R's print.FundamentalBooleanNetwork (diff with r_network.txt)
    reference_<name>.txt   the stored R-package network rendered with the same printer (with --compare-rda / a scenario)
    python_root_pool.csv   per target, the conditional genes that survive the first tree level
                           (cube[target]$SubGenes) - the earliest point at which two runs can diverge
    python_stage1_rules.csv  (only with --dump-stage1, ~60 MB) candidate rules BEFORE the per-gene cap of `max_fbn_rules`
    reproduce_in_R.R       the same experiment for R (writes r_rules.csv / r_network.txt / r_root_pool.csv, and r_stage1_rules.csv with --dump-stage1)
    comparison.txt         only with --compare-rda / --compare-csv: Python vs R rule-set comparison

Reference networks stored in the R package data (inspected with the pure-Python `rdata` package):
    FBM_Leukeamia_Networks.rda == fbm_leukeamia_network.rda : 2758 rules, timestep 1 only        -> temporal = 1
    Leukeamia_Networks.rda     == TFBM_Leukeamia_Networks.rda: 2775 rules, timesteps 1 and 2     -> temporal = 2
Both have up to 3 inputs per rule (max_k >= 3), error 0 and probability 1 on every rule.
The exact parameters R used to create those stored files are NOT recorded in the package, so the
scenarios below are inferred; `max_k` is the least certain one (see params.json `inferred_from`).
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from fbnnet_core.cube import construct_fbn_cube
from fbnnet_core.fbn_types import FundamentalBooleanNetwork
from fbnnet_core.network import mine_fbn_network, search_fbn_core
from fbnnet_core.network_app import _interaction_items

DATA_RDA = os.path.join(ROOT, "data", "Leukeamia_Timeseries.rda")

# Mining parameters of R's mineFBNNetwork()/constructFBNCube() defaults, spelled out so nothing is implicit.
FIXED_MINING_PARAMS = {
    "threshold_confidence": 1,
    "threshold_error": 0,
    "threshold_support": 1e-05,
    "max_fbn_rules": 5,
}
SCENARIOS = {
    "non_temporal": {"max_k": 3, "temporal": 1, "reference_rda": "FBM_Leukeamia_Networks",
                     "inferred_from": "reference has only timestep-1 rules (2758) with up to 3 inputs"},
    "temporal": {"max_k": 3, "temporal": 2, "reference_rda": "Leukeamia_Networks",
                 "inferred_from": "reference has timestep-1 and timestep-2 rules (2775) with up to 3 inputs"},
}


def load_leukaemia_timeseries(path=DATA_RDA):
    """Read the list of 26 sample matrices (285 genes x 2-3 time points, already 0/1) from the .rda file."""
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = rdata.conversion.convert(rdata.parser.parse_file(path))["Leukeamia_Timeseries"]
    samples = []
    for name, arr in raw.items():
        genes = [str(g) for g in arr.coords["dim_0"].values]
        cols = [str(c) for c in arr.coords["dim_1"].values]
        samples.append(pd.DataFrame(np.asarray(arr.values, dtype=float), index=genes, columns=cols))
    return [str(k) for k in raw.keys()], samples


def data_fingerprint(sample_names, samples):
    digest = hashlib.sha256()
    for name, df in zip(sample_names, samples):
        digest.update(name.encode())
        digest.update(",".join(df.index).encode())
        digest.update(",".join(df.columns).encode())
        digest.update(np.ascontiguousarray(df.to_numpy()).tobytes())
    return {
        "n_samples": len(samples),
        "n_genes": samples[0].shape[0],
        "timepoints_per_sample": {n: list(df.columns) for n, df in zip(sample_names, samples)},
        "all_values_binary": bool(all(set(np.unique(df.to_numpy())) <= {0.0, 1.0} for df in samples)),
        "sha256": digest.hexdigest(),
    }


def rules_table(network):
    rows = []
    for gene, inter in network["interactions"].items():
        for _, rule in _interaction_items(inter, gene):
            rows.append({
                "target": gene, "type": int(rule["type"]), "expression": rule["expression"],
                "timestep": int(rule["timestep"]), "support": rule["support"],
                "error": rule["error"], "probability": rule["probability"],
            })
    return pd.DataFrame(rows)


def write_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def root_pool_table(cube):
    rows = [{"target": t, "conditional_gene": g, "rank": i + 1}
            for t, node in cube["cube"].items() for i, g in enumerate(node["SubGenes"].keys())]
    return pd.DataFrame(rows)


def stage1_table(cube, genes, params):
    stage1 = search_fbn_core(cube["cube"], genes, False, params["threshold_confidence"],
                             params["threshold_error"], params["threshold_support"], params["max_fbn_rules"])
    rows = [{"target": t, "type": int(r["type"]), "input": r["input"], "numOfInput": int(r["numOfInput"]),
             "timestep": int(r["timestep"]), "support": r["support"], "error": r["error"]}
            for t, rules in stage1.items() for r in rules]
    return pd.DataFrame(rows)


R_SCRIPT = r'''# Generated by example/leukaemia_network_pipeline.py -- the same experiment as the Python run.
# Usage (R >= 4.0, package FBNNet 2.0.1 installed):  Rscript reproduce_in_R.R
suppressMessages(library(FBNNet))

timeseries <- NULL
data("Leukeamia_Timeseries")          # list of 26 matrices (285 genes x 2-3 time points), already 0/1
timeseries <- Leukeamia_Timeseries
genes <- rownames(timeseries[[1]])
stopifnot(length(timeseries) == {n_samples}, length(genes) == {n_genes})

maxK <- {max_k}; temporal <- {temporal}
cube <- constructFBNCube(target_genes = genes, conditional_genes = genes, timeseriesCube = timeseries,
                         maxK = maxK, temporal = temporal, useParallel = TRUE)

# 1) root level of the tree: conditional genes that survive the first level, per target
root <- do.call(rbind, lapply(names(cube), function(t) {{
  g <- names(cube[[t]]$SubGenes)
  if (length(g) == 0) return(NULL)
  data.frame(target = t, conditional_gene = g, rank = seq_along(g), stringsAsFactors = FALSE)
}}))
write.csv(root, "r_root_pool.csv", row.names = FALSE)

# 2) candidate rules before the per-gene cap
{stage1_block}
# 3) final network
network <- mineFBNNetwork(cube, genes, useParallel = FALSE,
                          threshold_confidence = {threshold_confidence}, threshold_error = {threshold_error},
                          threshold_support = {threshold_support}, maxFBNRules = {max_fbn_rules})
rows <- do.call(rbind, lapply(names(network$interactions), function(g) {{
  ints <- network$interactions[[g]]
  if (length(ints) == 0) return(NULL)
  do.call(rbind, lapply(ints, function(r) data.frame(
    target = g, type = as.numeric(r$type), expression = as.character(r$expression), timestep = as.numeric(r$timestep),
    support = as.numeric(r$support), error = as.numeric(r$error), probability = as.numeric(r$probability),
    stringsAsFactors = FALSE)))
}}))
write.csv(rows, "r_rules.csv", row.names = FALSE)
writeLines(capture.output(print(network)), "r_network.txt")
cat("R rules:", nrow(rows), " genes with rules:", length(unique(rows$target)), "\n")
print(sessionInfo())
'''


R_STAGE1_BLOCK = r'''stage1 <- search_FBN_core(cube, genes, FALSE, {threshold_confidence}, {threshold_error}, {threshold_support}, {max_fbn_rules})
s1 <- do.call(rbind, lapply(names(stage1), function(t) do.call(rbind, lapply(stage1[[t]], function(r)
  data.frame(target = t, type = as.numeric(r[["type"]]), input = r[["input"]], numOfInput = as.numeric(r[["numOfInput"]]),
             timestep = as.numeric(r[["timestep"]]), support = as.numeric(r[["support"]]), error = as.numeric(r[["error"]]),
             stringsAsFactors = FALSE)))))
write.csv(s1, "r_stage1_rules.csv", row.names = FALSE)'''


def compare_rule_tables(py_df, r_df, label):
    key = ["target", "type", "expression", "timestep"]
    p, r = set(map(tuple, py_df[key].values.tolist())), set(map(tuple, r_df[key].values.tolist()))
    common = p & r
    lines = [f"{label}: python rules={len(p)} reference rules={len(r)}",
             f"  identical (target,type,expression,timestep): {len(common)} = {len(common) / max(len(r), 1):.1%} of reference",
             f"  reference-only: {len(r - p)}   python-only: {len(p - r)}"]
    return "\n".join(lines)


def load_reference_rda(name):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = rdata.conversion.convert(rdata.parser.parse_file(os.path.join(ROOT, "data", f"{name}.rda")))[name]

    def scalar(v):
        v = v.tolist() if hasattr(v, "tolist") else v
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return v

    rows = []
    for tgt, inter in net["interactions"].items():
        items = inter.values() if hasattr(inter, "values") else inter
        for r in items:
            rows.append({"target": str(tgt), "type": int(float(scalar(r["type"]))), "expression": str(scalar(r["expression"])),
                         "timestep": int(float(scalar(r["timestep"]))), "support": float(scalar(r["support"]))})
    return pd.DataFrame(rows)


def load_reference_network(name):
    """The stored R-package network as a FundamentalBooleanNetwork, so it prints exactly like a mined one."""
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = rdata.conversion.convert(rdata.parser.parse_file(os.path.join(ROOT, "data", f"{name}.rda")))[name]

    def scalar(v):
        v = v.tolist() if hasattr(v, "tolist") else v
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return v

    genes = [str(g) for g in net["genes"]]
    interactions = {}
    for gene in genes:
        rules = net["interactions"][gene]
        rules = rules if hasattr(rules, "items") else {}  # a gene without rules comes back as an empty list
        interactions[gene] = {str(fn): {k: scalar(v) for k, v in r.items()} for fn, r in rules.items()}
    return FundamentalBooleanNetwork({
        "genes": genes, "interactions": interactions, "class": "FundamentalBooleanNetwork",
        "fixed": dict(zip(genes, [float(x) for x in np.ravel(net["fixed"])])),
        "timedecay": dict(zip(genes, [float(x) for x in np.ravel(net["timedecay"])])),
    })


def write_text(path, text):
    with open(path, "w") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")


def environment_info():
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(), "git_commit": commit}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), help="preset matching one of R's stored networks")
    ap.add_argument("--max-k", type=int, help="maximum tree depth (inputs per rule)")
    ap.add_argument("--temporal", type=int, help="maximum temporal step (1 = non-temporal)")
    ap.add_argument("--backend", choices=["recursive", "tensor"], default="recursive",
                    help="'recursive' = C++ engine (default, reference); 'tensor' = NumPy batching, validated identical")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: cpu_count-1)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--compare-rda", help="name of an R-package network to compare with, e.g. FBM_Leukeamia_Networks")
    ap.add_argument("--compare-csv", help="r_rules.csv written by reproduce_in_R.R, to compare with")
    ap.add_argument("--dump-stage1", action="store_true", help="also write the pre-cap candidate rules (~60 MB)")
    args = ap.parse_args()

    preset = SCENARIOS.get(args.scenario, {})
    max_k = args.max_k if args.max_k is not None else preset.get("max_k")
    temporal = args.temporal if args.temporal is not None else preset.get("temporal")
    if max_k is None or temporal is None:
        ap.error("give --scenario or both --max-k and --temporal")
    compare_rda = args.compare_rda or preset.get("reference_rda")
    out_dir = args.out_dir or os.path.join(HERE_OUT, args.scenario or f"k{max_k}_t{temporal}")
    os.makedirs(out_dir, exist_ok=True)
    logging.disable(logging.CRITICAL)

    sample_names, samples = load_leukaemia_timeseries()
    genes = list(samples[0].index)
    params = dict(FIXED_MINING_PARAMS, max_k=max_k, temporal=temporal)

    t0 = time.perf_counter()
    cube = construct_fbn_cube(genes, genes, samples, max_k=max_k, temporal=temporal, use_parallel=True,
                              max_workers=args.workers, backend=args.backend)
    t_cube = time.perf_counter() - t0
    t1 = time.perf_counter()
    network = mine_fbn_network(cube, genes=genes, use_parallel=False, **{k: params[k] for k in FIXED_MINING_PARAMS})
    t_mine = time.perf_counter() - t1

    py_rules = rules_table(network)
    write_csv(py_rules, os.path.join(out_dir, "python_rules.csv"))
    write_text(os.path.join(out_dir, "python_network.txt"), str(network))
    write_csv(root_pool_table(cube), os.path.join(out_dir, "python_root_pool.csv"))
    if args.dump_stage1:
        write_csv(stage1_table(cube, genes, params), os.path.join(out_dir, "python_stage1_rules.csv"))

    record = {
        "scenario": args.scenario, "inferred_from": preset.get("inferred_from"),
        "dataset": {"file": "data/Leukeamia_Timeseries.rda", "R_object": "Leukeamia_Timeseries",
                    "discretisation": "none (data already 0/1)", **data_fingerprint(sample_names, samples)},
        "cube": {"function": "constructFBNCube / construct_fbn_cube", "target_genes": "all", "conditional_genes": "all",
                 "maxK": max_k, "temporal": temporal, "backend": args.backend},
        "mining": {"function": "mineFBNNetwork / mine_fbn_network", "threshold_confidence": params["threshold_confidence"],
                   "threshold_error": params["threshold_error"], "threshold_support": params["threshold_support"],
                   "maxFBNRules": params["max_fbn_rules"], "rank_order_for_cap": "timestep, error, numOfInput, -support"},
        "result": {"genes": len(network["genes"]), "genes_with_rules": int(py_rules["target"].nunique()), "rules": len(py_rules),
                   "seconds_cube": round(t_cube, 1), "seconds_mining": round(t_mine, 1)},
        "environment": environment_info(),
    }
    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    with open(os.path.join(out_dir, "reproduce_in_R.R"), "w") as fh:
        stage1_block = R_STAGE1_BLOCK.format(**params) if args.dump_stage1 else "# (re-run with --dump-stage1 to also export the pre-cap candidate rules)"
        fh.write(R_SCRIPT.replace("{stage1_block}", stage1_block).format(n_samples=len(samples), n_genes=len(genes), max_k=max_k, temporal=temporal,
                                 threshold_confidence=params["threshold_confidence"], threshold_error=params["threshold_error"],
                                 threshold_support=params["threshold_support"], max_fbn_rules=params["max_fbn_rules"]))

    report = []
    if args.compare_csv:
        report.append(compare_rule_tables(py_rules, pd.read_csv(args.compare_csv), f"vs R run ({args.compare_csv})"))
    if compare_rda:
        report.append(compare_rule_tables(py_rules, load_reference_rda(compare_rda), f"vs stored {compare_rda}.rda"))
        write_text(os.path.join(out_dir, f"reference_{compare_rda}.txt"), str(load_reference_network(compare_rda)))
    if report:
        with open(os.path.join(out_dir, "comparison.txt"), "w") as fh:
            fh.write("\n".join(report) + "\n")
    print(json.dumps(record["result"]), "\n" + "\n".join(report), f"\nwritten to {out_dir}")


HERE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leukaemia_output")

if __name__ == "__main__":
    main()
