#!/usr/bin/env python
"""Benchmark of the whole mining pipeline: cube first, then rules ("cube") versus rules mined per target
as each cube entry is built ("stream", `mine_fbn_network_from_timeseries`).

Both variants run on the same randomly chosen targets and must return the identical network. Each
variant runs in a fresh process so that the peak memory of the parent process is comparable.

Takes the data options and `--targets/--seed/--max-k/--temporal/--workers` of benchmark_cube.py, e.g.

  python tools/benchmark_pipeline.py --data npz --npz data/testdata/derived/GSE2677_genes_8000.npz --targets 64
  python tools/benchmark_pipeline.py --data synthetic --genes 2000 --targets 32 --sequential
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
if (TOOLS.parent / "fbnnet_core" / "__init__.py").exists():
    sys.path.insert(0, str(TOOLS.parent))  # a source checkout (with built extensions) wins over an installed copy
import benchmark_cube as bench  # noqa: E402

VARIANTS = ("cube", "stream")


def build_parser():
    parser = bench.build_parser()
    parser.description = __doc__
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--sequential", action="store_true", help="run without worker processes")
    parser.add_argument("--result", help=argparse.SUPPRESS)
    return parser


def choose_targets(args, genes: List[str]) -> List[str]:
    rng = np.random.default_rng(args.seed)
    n_targets = min(args.targets, len(genes))
    return [genes[i] for i in sorted(rng.choice(len(genes), size=n_targets, replace=False))]


def run_variant(args) -> Dict[str, Any]:
    from fbnnet_core import _tree
    from fbnnet_core.cube import DEFAULT_MAX_WORKERS, construct_fbn_cube, convert_df_main_parameters
    from fbnnet_core.network import mine_fbn_network, mine_fbn_network_from_timeseries

    series = bench.load_data(args)
    genes = list(series[0].index)
    targets = choose_targets(args, genes)
    main_parameters = convert_df_main_parameters(series, args.temporal)
    targets = _tree.filterTargetGenesByConditionGenes(targets, main_parameters, genes, None, args.temporal)
    del main_parameters
    parallel = {
        "use_parallel": not args.sequential,
        "max_workers": args.workers or DEFAULT_MAX_WORKERS,
    }
    started = time.perf_counter()
    if args.variant == "cube":
        cube = construct_fbn_cube(
            targets, genes, series, max_k=args.max_k, temporal=args.temporal, backend="tensor", **parallel
        )
        cube_done = time.perf_counter() - started
        cube_bytes = len(pickle.dumps(cube["cube"], protocol=pickle.HIGHEST_PROTOCOL))
        network = mine_fbn_network(cube, genes=targets, **parallel)
    else:
        cube_done, cube_bytes = None, None
        network = mine_fbn_network_from_timeseries(
            genes, series, max_k=args.max_k, temporal=args.temporal, backend="tensor",
            target_genes=targets, **parallel,
        )
    wall = time.perf_counter() - started
    digest = hashlib.sha256(pickle.dumps(network, protocol=4)).hexdigest()
    return {
        "variant": args.variant, "targets": len(targets), "genes": len(genes), "wall": wall,
        "cube_stage": cube_done, "cube_pickled_mb": None if cube_bytes is None else cube_bytes / 1e6,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "interactions": sum(len(v) if hasattr(v, "__len__") else 1 for v in network["interactions"].values()),
        "network_sha": digest,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.variant:
        result = run_variant(args)
        Path(args.result).write_text(json.dumps(result))
        return 0

    raw = list(sys.argv[1:] if argv is None else argv)
    results = []
    for variant in VARIANTS:
        result_file = Path(f"/tmp/benchmark_pipeline_{os.getpid()}_{variant}.json")
        print(f"running {variant} ...", flush=True)
        subprocess.run(
            [sys.executable, __file__, *raw, "--variant", variant, "--result", str(result_file)], check=True
        )
        results.append(json.loads(result_file.read_text()))
        result_file.unlink()
    cube, stream = results
    same = cube["network_sha"] == stream["network_sha"]
    print(f"\ngenes={cube['genes']} targets={cube['targets']} workers="
          f"{'1 (sequential)' if args.sequential else args.workers or 'default'}")
    print(f"{'pipeline':8s} {'wall s':>8s} {'parent peak MB':>15s} {'cube pickled MB':>16s}  identical network")
    for r in results:
        pickled = "-" if r["cube_pickled_mb"] is None else f"{r['cube_pickled_mb']:.0f}"
        print(f"{r['variant']:8s} {r['wall']:8.2f} {r['peak_rss_mb']:15.0f} {pickled:>16s}  "
              f"{'yes' if same else 'NO'} ({r['interactions']} interactions)")
    print(f"speedup {cube['wall'] / stream['wall']:.1f}x, parent memory {cube['peak_rss_mb'] / stream['peak_rss_mb']:.1f}x lower")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
