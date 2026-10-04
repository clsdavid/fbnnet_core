"""
Reproduces the worked example from the original R package's vignette
(FBNNet2_public/vignettes/FBNNet.Rmd, section "Extract the Fundamental
Boolean Network") using this Python port end to end:

  1. Load the BoolNet-style network `example.bn` (identical to the R
     package's `ExampleNetwork` test fixture,
     FBNNet2_public/tests/testthat/example.bn).
  2. Generate every possible initial state for the 5 genes.
  3. Simulate 43 synchronous BoolNet time steps from each initial state
     (`generateBoolNetTimeseries`, mirroring the R vignette's
     `genereateBoolNetTimeseries(ExampleNetwork, initialStates, 43,
     type = "synchronous")`).
  4. Mine a Fundamental Boolean Network from that data in one call
     (`generate_fbm_network`, the port of R's `generateFBMNetwork`).
  5. Render the mined network (gene -> gene, and a zoomed-in
     gene -> rule -> gene view) to PNGs.
  6. Search for attractors of the mined FBN and render the first one found.

Note: `get_fbm_successor` breaks probability ties between competing
activator/inhibitor rules at random (this mirrors the "uncertainty"
mechanism described in the FBM paper), so a fixed random seed is used here
to make the attractor-search results below reproducible run to run.

Run from the repository root with the project's virtualenv active:

    python example/reproduce_vignette_experiment.py

Outputs are written next to this script:
    example/mined_network_summary.txt
    example/mined_network_graph.png
    example/mined_network_graph_gene1_forward.png
    example/attractor_0.png
"""
import os
import random

import pandas as pd

from fbnnet_core.boolnet import load_network
from fbnnet_core.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
from fbnnet_core.application import generate_fbm_network
from fbnnet_core.attractor import search_for_attractors
from fbnnet_core.network_graph import fbn_network_graph, plot_network, draw_attractor

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. Load the BoolNet network.
    network = load_network(os.path.join(HERE, "example.bn"))
    genes = network["genes"]
    print(f"Loaded network with genes: {genes}")
    print(f"network: {network}")

    # 2 & 3. Simulate synchronous BoolNet time series from every initial state.
    initial_states = generateAllCombinationBinary(genes)
    # the method `generateBoolNetTimeseries` should closely mirror the R vignette's `genereateBoolNetTimeseries` function.
    raw_series = generateBoolNetTimeseries(
        network, initial_states, numMeasurements=43, transition_type="synchronous"
    )
    trainingseries = [
        pd.DataFrame(mat, index=genes, columns=[str(j + 1) for j in range(mat.shape[1])])
        for mat in raw_series
    ]
    print(f"Simulated {len(trainingseries)} time series of shape {trainingseries[0].shape}")

    # 4. Mine the Fundamental Boolean Network.
    fbn_network = generate_fbm_network(
        trainingseries,
        max_k=4,
        max_deep_temporal=1,
        use_parallel=False,
        network_only=True,
        verbose=False,
    )
    print(f"Generated FBN network: {fbn_network}")
    summary_lines = ["Mined FBN network interactions:"]
    for gene, rules in fbn_network["interactions"].items():
        for name, rule in (rules.items() if isinstance(rules, dict) else enumerate(rules)):
            summary_lines.append(
                f"  {gene} <- {rule['expression']} "
                f"(type={'activator' if rule['type'] == 1 else 'inhibitor'}, "
                f"timestep={rule['timestep']}, probability={rule['probability']:.3f})"
            )
    print("\n".join(summary_lines))

    summary_path = os.path.join(HERE, "mined_network_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"Saved mined network summary to {summary_path}")

    # 5. Render the mined network (genes, activator/inhibitor rules and their inputs), plus a
    # zoomed-in view of what Gene1 regulates.
    fbn_network_graph(fbn_network).save(os.path.join(HERE, "mined_network_graph.png"), dpi=150)
    print(f"Saved network graph to {os.path.join(HERE, 'mined_network_graph.png')}")

    plot_network(fbn_network, target_genes=["Gene1"], type="forward_1a", expand_level=1).save(
        os.path.join(HERE, "mined_network_graph_gene1_forward.png"), dpi=150
    )
    print(f"Saved Gene1 forward rule-graph to {os.path.join(HERE, 'mined_network_graph_gene1_forward.png')}")

    # 6. Search for attractors (seeded for reproducibility) and render the first one.
    random.seed(42)
    attractors = search_for_attractors(fbn_network, genes, transition_type="synchronous")
    cycles = attractors["Attractors"]
    basins = attractors["BasinOfAttractor"]
    print(f"Found {len(cycles)} attractor(s); cycle lengths: {[len(c) for c in cycles]}; "
          f"basin sizes: {[len(b) for b in basins]}")

    draw_attractor(fbn_network, attractors, index=0).save(os.path.join(HERE, "attractor_0.png"), dpi=150)
    print(f"Saved attractor drawing to {os.path.join(HERE, 'attractor_0.png')}")


if __name__ == "__main__":
    main()
