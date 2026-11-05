#!/usr/bin/env python
"""Benchmark harness for `construct_fbn_cube` (Phase A of .temp/next_generation_improvement_plan.md).

It runs the Orchard Cube construction with the sequential/parallel x recursive/tensor
configurations on the same randomly chosen target genes, checks that every configuration
returns the identical cube, and reports

* wall time, speedup and parallel efficiency per configuration;
* survivor statistics per target (S = genes that survive the root-level test, S/G, tree nodes);
* the per-target wall time distribution (sequential configurations);
* a fit of the cost model  time ~ a + b * S^2  and an extrapolated time for a full run.

Data sources (`--data`):
  synthetic   random 0/1 time series with optional planted regulation (--genes, --samples, ...)
  leukaemia   the bundled 285-gene real data set (Leukeamia_Timeseries)
  npz         time series written by fbnnet_core.cel_import (--npz FILE)
  cel         Affymetrix CEL files converted on the fly (--cel-dir DIR --cdf FILE [--n-genes N])

Examples:
  python tools/benchmark_cube.py --data synthetic --genes 2000 --samples 20 --timepoints 50 --targets 32
  python tools/benchmark_cube.py --data leukaemia --targets 40 --configs tensor-seq recursive-seq
  python tools/benchmark_cube.py --data cel --cel-dir data/testdata/GSE2677_RAW \\
      --cdf data/testdata/HG-U133_Plus_2/LibFiles/HG-U133_Plus_2.cdf --n-genes 8000 \\
      --targets 16 --configs tensor-seq tensor-par --json results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CONFIGS = ("tensor-seq", "tensor-par", "recursive-seq", "recursive-par")


# --------------------------------------------------------------------------- data

def make_synthetic(
    genes: int = 2000,
    samples: int = 20,
    timepoints: int = 50,
    density: float = 0.4,
    planted: int = 0,
    noise: float = 0.05,
    seed: int = 0,
) -> List[pd.DataFrame]:
    """Random 0/1 time series (genes x timepoints, one table per sample).

    Every value is 1 with probability `density`. For `planted` genes the value at t+1 is set to
    ``a(t) AND NOT b(t)`` of two other genes (with `noise` flips), which creates real rules whose
    survivors the mining then has to find.
    """
    rng = np.random.default_rng(seed)
    names = [f"G{i:05d}" for i in range(genes)]
    data = (rng.random((samples, genes, timepoints)) < density).astype(np.int8)
    if planted:
        chosen = rng.choice(genes, size=min(planted, genes), replace=False)
        for target in chosen:
            a, b = rng.choice(genes, size=2, replace=False)
            rule = data[:, a, :-1] & (1 - data[:, b, :-1])
            flips = rng.random(rule.shape) < noise
            data[:, target, 1:] = np.where(flips, 1 - rule, rule)
    columns = [str(t + 1) for t in range(timepoints)]
    return [pd.DataFrame(data[s], index=names, columns=columns) for s in range(samples)]


def load_data(args: argparse.Namespace) -> List[pd.DataFrame]:
    if args.data == "synthetic":
        return make_synthetic(args.genes, args.samples, args.timepoints, args.density, args.planted, args.noise, args.seed)
    if args.data == "leukaemia":
        from fbnnet_core import load_dataset

        return load_dataset("Leukeamia_Timeseries")
    if args.data == "npz":
        from fbnnet_core.cel_import import load_timeseries

        return load_timeseries(args.npz)
    if args.data == "cel":
        from fbnnet_core.cel_import import cel_to_timeseries

        return cel_to_timeseries(args.cel_dir, args.cdf, n_genes=args.n_genes, cache_path=args.cache)
    raise ValueError(f"unknown data source {args.data}")


# --------------------------------------------------------------------------- statistics

def tree_nodes(genes: Dict[str, Any]) -> int:
    """Number of rule nodes below a dict of genes (the T and F branches are counted recursively)."""
    total = 0
    for entry in genes.values():
        total += 1
        if isinstance(entry, dict):
            total += tree_nodes(entry.get("SubGenesT", {})) + tree_nodes(entry.get("SubGenesF", {}))
    return total


def survivor_stats(
    cube: Dict[str, Any], n_candidates: int, targets: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Per target: S (root-level survivors), S/G and the number of nodes in the whole tree.

    Targets without survivors are absent from the cube; pass `targets` to count them with S = 0.
    """
    stats = {}
    for target in (targets if targets is not None else cube):
        entry = cube.get(target)
        root = entry.get("SubGenes", {}) if isinstance(entry, dict) else {}
        stats[target] = {"S": len(root), "S_over_G": len(root) / n_candidates, "nodes": tree_nodes(root)}
    return stats


def distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {
        "min": ordered[0], "median": statistics.median(ordered), "mean": statistics.fmean(ordered),
        "p90": p90, "max": ordered[-1],
    }


def human_duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def parallel_lower_bound(times: Sequence[float], workers: int) -> float:
    """Best possible wall time on `workers` processes: no faster than the average share or the longest target."""
    return max(sum(times) / max(1, workers), max(times)) if times else 0.0


def fit_linear(x: Sequence[float], y: Sequence[float]) -> Optional[Dict[str, float]]:
    """Least-squares y = a + b*x with R^2; None when x has no spread."""
    x_arr, y_arr = np.asarray(x, float), np.asarray(y, float)
    if len(x_arr) < 3 or np.ptp(x_arr) == 0:
        return None
    design = np.column_stack([np.ones_like(x_arr), x_arr])
    (a, b), *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    residual = y_arr - design @ np.array([a, b])
    total = ((y_arr - y_arr.mean()) ** 2).sum()
    return {"a": float(a), "b": float(b), "r2": float(1 - (residual ** 2).sum() / total) if total > 0 else 1.0}


# --------------------------------------------------------------------------- runs

def run_config(
    name: str,
    targets: List[str],
    candidates: List[str],
    max_k: int,
    temporal: int,
    main_parameters: Any,
    workers: int,
    chunksize: Optional[int],
    timeseries_cube: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Runs one configuration; sequential ones also return the wall time of every target."""
    from fbnnet_core import _parallel
    from fbnnet_core import cube as cube_module

    backend, mode = name.split("-")
    started = time.perf_counter()
    per_target: Dict[str, float] = {}
    if mode == "seq":
        process = cube_module._process_cube_algorithm_for_backend(backend)
        results = []
        for gene in targets:
            t0 = time.perf_counter()
            results.append(process(gene, candidates, max_k, temporal, main_parameters, None, None))
            per_target[gene] = time.perf_counter() - t0
        used_workers = 1
    else:
        results = cube_module.do_parallel_work(
            targets, candidates, max_k, temporal, main_parameters, workers, chunksize, backend, timeseries_cube
        )
        method = _parallel.start_method()
        spawn_without_data = method == "spawn" and timeseries_cube is None
        used_workers = max(1, min(workers, len(targets))) if method and not spawn_without_data else 1
    wall = time.perf_counter() - started
    cube: Dict[str, Any] = {}
    for item in results:
        if item:
            cube.update(item)
    return {"name": name, "backend": backend, "mode": mode, "workers": used_workers, "wall": wall,
            "per_target": per_target, "cube": cube, "targets": list(targets)}


def compare_cubes(reference: Dict[str, Any], other: Dict[str, Any]) -> Tuple[bool, int]:
    """(identical, number of targets compared) on the targets both cubes contain."""
    common = [t for t in reference if t in other]
    return all(reference[t] == other[t] for t in common), len(common)


def benchmark(args: argparse.Namespace, out=sys.stdout) -> Dict[str, Any]:
    from fbnnet_core import _tree
    from fbnnet_core.cube import DEFAULT_MAX_WORKERS, convert_df_main_parameters

    def say(text: str = "") -> None:
        if not args.quiet:
            print(text, file=out, flush=True)

    series = load_data(args)
    genes = list(series[0].index)
    rng = np.random.default_rng(args.seed)
    n_targets = min(args.targets, len(genes))
    targets = [genes[i] for i in sorted(rng.choice(len(genes), size=n_targets, replace=False))]
    workers = args.workers or DEFAULT_MAX_WORKERS
    timepoints = series[0].shape[1]
    say(f"data: {args.data}  genes={len(genes)} samples={len(series)} timepoints={timepoints}  "
        f"density={float(np.mean([s.to_numpy().mean() for s in series])):.3f}")
    say(f"targets={n_targets} (seed {args.seed}) max_k={args.max_k} temporal={args.temporal} workers={workers}  "
        f"python {platform.python_version()} numpy {np.__version__} cpus={os.cpu_count()}")

    started = time.perf_counter()
    main_parameters = convert_df_main_parameters(series, args.temporal)
    targets = _tree.filterTargetGenesByConditionGenes(targets, main_parameters, genes, None, args.temporal)
    setup = time.perf_counter() - started
    say(f"setup (matrices, target filter): {setup:.2f}s; {len(targets)} targets kept")

    runs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name in args.configs:
        run_targets = targets
        if name.startswith("recursive"):
            if args.max_recursive_genes and len(genes) > args.max_recursive_genes:
                skipped.append(name)
                say(f"skip {name}: {len(genes)} genes > --max-recursive-genes {args.max_recursive_genes}")
                continue
            if args.recursive_targets:
                run_targets = targets[: args.recursive_targets]
        say(f"running {name} on {len(run_targets)} targets ...")
        repeats = [run_config(name, run_targets, genes, args.max_k, args.temporal, main_parameters, workers, args.chunksize, series)
                   for _ in range(max(1, args.repeat))]
        run = repeats[0]
        run["walls"] = [r["wall"] for r in repeats]
        run["wall"] = statistics.median(run["walls"])  # the first repetition keeps the per-target times
        say("  " + (f"{run['wall']:.2f}s" if len(repeats) == 1 else
                    f"median {run['wall']:.2f}s of " + ", ".join(f"{w:.2f}" for w in run["walls"])))
        runs.append(run)

    report: Dict[str, Any] = {
        "data": {"source": args.data, "genes": len(genes), "samples": len(series), "timepoints": timepoints},
        "targets": targets, "max_k": args.max_k, "temporal": args.temporal, "workers": workers,
        "setup_seconds": setup, "skipped": skipped, "configs": [],
    }
    if not runs:
        return report

    baseline = runs[0]
    for run in runs:
        same, compared = compare_cubes(baseline["cube"], run["cube"])
        # speedup against the same engine's sequential run (or the baseline) on the same targets
        same_engine_seq = next((r for r in runs if r["backend"] == run["backend"] and r["mode"] == "seq"), None)
        reference = same_engine_seq or baseline
        ref_wall = reference["wall"] if reference["targets"] == run["targets"] else None
        if ref_wall is None and reference["per_target"]:
            ref_wall = sum(reference["per_target"].get(t, 0.0) for t in run["targets"]) or None
        run.update({
            "identical_to_first": same, "compared_targets": compared,
            "speedup_vs_seq": (ref_wall / run["wall"]) if ref_wall else None,
        })
        run["efficiency"] = (run["speedup_vs_seq"] / run["workers"]) if (run["mode"] == "par" and run["speedup_vs_seq"]) else None
        report["configs"].append({k: v for k, v in run.items() if k not in ("cube", "per_target", "targets")}
                                 | {"n_targets": len(run["targets"])})

    say()
    header = f"{'config':<16}{'targets':>8}{'workers':>8}{'wall s':>10}{'s/target':>10}{'speedup':>9}{'effic.':>8}  identical"
    say(header)
    say("-" * len(header))
    for run, summary in zip(runs, report["configs"]):
        speedup = f"{summary['speedup_vs_seq']:.1f}x" if summary["speedup_vs_seq"] else "-"
        efficiency = f"{summary['efficiency']:.0%}" if summary["efficiency"] else "-"
        say(f"{run['name']:<16}{len(run['targets']):>8}{run['workers']:>8}{run['wall']:>10.2f}"
            f"{run['wall'] / len(run['targets']):>10.3f}{speedup:>9}{efficiency:>8}  "
            f"{'yes' if summary['identical_to_first'] else 'NO'} ({summary['compared_targets']})")
    if not all(c["identical_to_first"] for c in report["configs"]):
        say("WARNING: cubes differ between configurations")

    # A2: survivor statistics and per-target time distribution (from the first run of each engine)
    report["survivors"] = {}
    report["per_target_time"] = {}
    report["cost_model"] = {}
    for run in runs:
        if run["mode"] != "seq" or run["backend"] in report["per_target_time"]:
            continue
        stats = survivor_stats(run["cube"], len(genes), run["targets"])
        times = [run["per_target"][t] for t in run["targets"] if t in run["per_target"]]
        s_values = [stats[t]["S"] for t in run["targets"] if t in stats]
        node_values = [stats[t]["nodes"] for t in run["targets"] if t in stats]
        report["survivors"][run["backend"]] = {
            "S": distribution(s_values),
            "S_over_G": distribution([stats[t]["S_over_G"] for t in run["targets"] if t in stats]),
            "nodes": distribution(node_values),
            "per_target": stats,
        }
        report["per_target_time"][run["backend"]] = {
            "seconds": distribution(times), "per_target": run["per_target"], "n_targets": len(times),
            "parallel_lower_bound_seconds": parallel_lower_bound(times, workers),
        }
        fit_s2 = fit_linear([s * s for s in s_values], times)
        fit_nodes = fit_linear(node_values, times)
        mean_time = statistics.fmean(times) if times else 0.0
        total = args.extrapolate_targets or len(genes)
        report["cost_model"][run["backend"]] = {
            "fit_time_vs_S2": fit_s2, "fit_time_vs_nodes": fit_nodes,
            "mean_seconds_per_target": mean_time,
            "extrapolated_targets": total,
            "extrapolated_seconds_with_workers": mean_time * total / max(1, workers),
        }

    for backend, survivors in report["survivors"].items():
        say()
        say(f"[{backend}] survivors per target (G = {len(genes)} candidates, {report['per_target_time'][backend]['n_targets']} targets)")
        s, ratio, nodes = survivors["S"], survivors["S_over_G"], survivors["nodes"]
        say(f"  S      min {s['min']:.0f}  median {s['median']:.0f}  mean {s['mean']:.1f}  p90 {s['p90']:.0f}  max {s['max']:.0f}")
        say(f"  S/G    min {ratio['min']:.2%}  median {ratio['median']:.2%}  mean {ratio['mean']:.2%}  max {ratio['max']:.2%}")
        say(f"  nodes  min {nodes['min']:.0f}  median {nodes['median']:.0f}  mean {nodes['mean']:.1f}  max {nodes['max']:.0f}")
        t = report["per_target_time"][backend]["seconds"]
        say(f"  time/target (s)  min {t['min']:.3f}  median {t['median']:.3f}  mean {t['mean']:.3f}  "
            f"p90 {t['p90']:.3f}  max {t['max']:.3f}")
        model = report["cost_model"][backend]
        bound = report["per_target_time"][backend]["parallel_lower_bound_seconds"]
        longest = report["per_target_time"][backend]["seconds"]["max"]
        say(f"  parallel lower bound on {workers} workers: {human_duration(bound)} "
            f"(longest target {human_duration(longest)}, mean share {human_duration(t['mean'] * report['per_target_time'][backend]['n_targets'] / workers)})")
        fit = model["fit_time_vs_S2"]
        if fit:
            say(f"  fit  time = {fit['a']:.4f} + {fit['b'] * 1e6:.2f}e-6 * S^2   (R^2 {fit['r2']:.2f})")
        fit = model["fit_time_vs_nodes"]
        if fit:
            say(f"  fit  time = {fit['a']:.4f} + {fit['b'] * 1e3:.3f}e-3 * nodes (R^2 {fit['r2']:.2f})")
        say(f"  extrapolation: {model['extrapolated_targets']} targets x {model['mean_seconds_per_target']:.3f}s "
            f"/ {workers} workers = {human_duration(model['extrapolated_seconds_with_workers'])}")
    return report


# --------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["synthetic", "leukaemia", "npz", "cel"], default="synthetic")
    synthetic = parser.add_argument_group("synthetic data")
    synthetic.add_argument("--genes", type=int, default=2000)
    synthetic.add_argument("--samples", type=int, default=20)
    synthetic.add_argument("--timepoints", type=int, default=50)
    synthetic.add_argument("--density", type=float, default=0.4)
    synthetic.add_argument("--planted", type=int, default=0, help="number of genes with a planted rule")
    synthetic.add_argument("--noise", type=float, default=0.05, help="flip probability of planted rules")
    real = parser.add_argument_group("real data")
    real.add_argument("--npz", help="time series written by fbnnet_core.cel_import")
    real.add_argument("--cel-dir")
    real.add_argument("--cdf")
    real.add_argument("--n-genes", type=int, default=0, help="most variable genes to keep (cel)")
    real.add_argument("--cache", help="expression cache for --data cel")
    run = parser.add_argument_group("run")
    run.add_argument("--targets", type=int, default=16, help="randomly chosen target genes")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-k", type=int, default=2)
    run.add_argument("--temporal", type=int, default=1)
    run.add_argument("--configs", nargs="+", choices=CONFIGS, default=["tensor-seq", "tensor-par", "recursive-seq", "recursive-par"])
    run.add_argument("--workers", type=int, default=None, help="parallel workers (default cpu_count - 1)")
    run.add_argument("--repeat", type=int, default=1,
                     help="repeat every configuration and report the median wall time (timings are noisy)")
    run.add_argument("--chunksize", type=int, default=None)
    run.add_argument("--max-recursive-genes", type=int, default=4000,
                     help="skip recursive configurations above this many genes (0 = never skip)")
    run.add_argument("--recursive-targets", type=int, default=0,
                     help="run the recursive configurations on only the first N targets (0 = all)")
    run.add_argument("--extrapolate-targets", type=int, default=0,
                     help="targets of the extrapolated full run (default: all genes)")
    out = parser.add_argument_group("output")
    out.add_argument("--json", help="write the full report (incl. per-target data) to this file")
    out.add_argument("--csv", help="write the per-target survivors and times to this file")
    out.add_argument("--quiet", action="store_true")
    return parser


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.data == "npz" and not args.npz:
        parser.error("--data npz needs --npz")
    if args.data == "cel" and not (args.cel_dir and args.cdf):
        parser.error("--data cel needs --cel-dir and --cdf")
    if args.targets < 1:
        parser.error("--targets must be positive")


def write_outputs(report: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, default=float)
    if args.csv and report.get("survivors"):
        rows = []
        for backend, survivors in report["survivors"].items():
            for target, values in survivors["per_target"].items():
                rows.append({"backend": backend, "target": target, **values,
                             "seconds": report["per_target_time"][backend]["per_target"].get(target)})
        pd.DataFrame(rows).to_csv(args.csv, index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)
    report = benchmark(args)
    write_outputs(report, args)
    identical = all(c["identical_to_first"] for c in report["configs"])
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
