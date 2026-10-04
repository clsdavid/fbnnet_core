"""Regenerate the figures used in README.md (docs/images/*.png).

Run from the repository root: python tools/make_readme_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")

from fbnnet_core import load_dataset, search_for_attractors  # noqa: E402
from fbnnet_core.network_app import find_all_backward_related_genes, find_forward_related_network_by_genes  # noqa: E402
from fbnnet_core.network_graph import draw_attractor, fbn_network_graph  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images")
DPI = 110


def save(graph, name):
    if graph is None:
        print(f"{name}: empty network, skipped")
        return
    path = os.path.join(OUT, name)
    graph.save(path, dpi=DPI)
    print(f"{name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {os.path.getsize(path) // 1024} KB")


def main():
    os.makedirs(OUT, exist_ok=True)
    target = "CDC42EP3"
    for prefix, dataset in (("tfbm", "TFBM_Leukeamia_Networks"), ("fbm", "FBM_Leukeamia_Networks")):
        networks = load_dataset(dataset)
        forward = dict(networks=networks, target_gene_list=[target])
        save(fbn_network_graph(find_forward_related_network_by_genes(
            **forward, regulation_type=1, target_type=1, max_deep=1)), f"{prefix}_forward_activating_activated.png")
        save(fbn_network_graph(find_forward_related_network_by_genes(
            **forward, regulation_type=0, target_type=1, max_deep=1)), f"{prefix}_forward_inhibiting_activated.png")
        save(fbn_network_graph(find_forward_related_network_by_genes(
            **forward, regulation_type=1, target_type=0, max_deep=2, next_level_mix_type=True)),
            f"{prefix}_forward_two_levels.png")
        save(fbn_network_graph(find_all_backward_related_genes(
            networks, target, regulation_type=0, target_type=1, max_deep=1)), f"{prefix}_backward.png")

    network = load_dataset("FBNExampleNetworks")
    series = load_dataset("ExampleTimeseriesData")[0]
    save(fbn_network_graph(network), "example_static.png")
    save(fbn_network_graph(network, type="staticSlice", timeseries_matrix=series, to_time_point=3), "example_slice.png")
    save(fbn_network_graph(network, type="dynamic", timeseries_matrix=series, from_time_point=1, to_time_point=4),
         "example_dynamic.png")
    attractors = search_for_attractors(network, network["genes"])
    longest = max(range(len(attractors["Attractors"])), key=lambda i: len(attractors["Attractors"][i]))
    save(draw_attractor(network, attractors, longest), "example_attractor.png")


if __name__ == "__main__":
    main()
