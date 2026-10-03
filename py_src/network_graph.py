"""
Network visualisation helpers.

A pragmatic Python port of R's `graph_FBN.R` / `plot_network_FBN.R`. Rather than
reproducing the exact visNetwork/igraph JSON structures those files build (which
target an interactive JS widget with no direct Python equivalent), this module
converts an FBN network into a `networkx` graph and renders it with
`matplotlib`, covering the same conceptual operations:

* `to_networkx_graph`          -- port of `ConvertToNetworkGraphicObject`
* `draw_static_network`        -- port of `FBNNetwork.Graph(type = "static")` / `StaticNetwork`
* `draw_static_network_slice`  -- port of `FBNNetwork.Graph(type = "staticSlice")` / `StaticNetworkInSlice`
* `draw_dynamic_network`       -- port of `FBNNetwork.Graph(type = "dynamic")` / `GenerateDynamicNetworkGraphicObject`
* `plot_network`               -- port of `plotNetwork` (filters, then draws)
* `draw_attractor`             -- port of `FBNNetwork.Graph.DrawAttractor`
"""
import logging
import os
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import fbnnet_utils

# Use a non-interactive backend by default so this module works headlessly
# (e.g. in CI/tests); callers can override MPLBACKEND before import if they
# need an interactive backend.
os.environ.setdefault("MPLBACKEND", "Agg")

from .network_app import (
    _interaction_items,
    filter_network_connections_by_genes,
    find_forward_related_network_by_genes,
    find_backward_related_network_by_genes,
)

logger = logging.getLogger(__name__)

ACTIVATOR_COLOR = "darkgreen"
INHIBITOR_COLOR = "darkred"
DECAY_COLOR = "grey"

_FBN_NETWORK_CLASSES = ("FundamentalBooleanNetwork", "BooleanNetworkCollection")


def to_networkx_graph(fbn_network: Dict[str, Any], show_decay: bool = False) -> "nx.MultiDiGraph":
    """
    Convert an FBN network into a `networkx.MultiDiGraph`.

    Port of R's `ConvertToNetworkGraphicObject`. Node/edge styling metadata
    ("color", "style") is stored as edge attributes so a renderer can use it
    directly.

    Args:
        fbn_network: The Fundamental Boolean Network
        show_decay: If True, add a self-loop "decay" edge for every gene

    Returns:
        A `networkx.MultiDiGraph` with one node per gene and one edge per
        activator/inhibitor connection (plus decay self-loops if requested).
    """
    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    genes = fbn_network["genes"]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(genes)

    interactions = fbn_network.get("interactions", {})
    for target_gene in genes:
        interactions_for_gene = interactions.get(target_gene, {})

        if show_decay:
            graph.add_edge(target_gene, target_gene, kind="decay", color=DECAY_COLOR, style="dashed")

        for _, interaction in _interaction_items(interactions_for_gene, target_gene):
            interaction_type = interaction.get("type", 1)
            for idx in interaction.get("input", []):
                source_gene = genes[idx - 1]
                graph.add_edge(
                    source_gene,
                    target_gene,
                    kind="activator" if interaction_type == 1 else "inhibitor",
                    color=ACTIVATOR_COLOR if interaction_type == 1 else INHIBITOR_COLOR,
                    style="solid",
                    timestep=interaction.get("timestep", 1),
                    probability=interaction.get("probability"),
                    expression=interaction.get("expression"),
                )

    return graph


def to_networkx_graph_with_rules(fbn_network: Dict[str, Any], show_decay: bool = False) -> "nx.MultiDiGraph":
    """
    Convert an FBN network into a gene -> rule -> gene `networkx.MultiDiGraph`.

    Unlike `to_networkx_graph` (which collapses each rule into a single direct
    gene->gene edge), this mirrors R's `ConvertToNetworkGraphicObject` more
    closely: every individual activator/inhibitor rule gets its own "rule"
    node (labelled with its type and timestep), with edges
    `input gene -> rule -> target gene`. Input edges are colored red if that
    specific input is negated (`!gene`) inside the rule's expression, green
    otherwise -- matching R's per-input negation highlighting.

    Args:
        fbn_network: The Fundamental Boolean Network
        show_decay: If True, add a self-loop "decay" edge for every gene

    Returns:
        A `networkx.MultiDiGraph` with `node_kind` in {"gene", "rule"} node
        attributes, one rule node per interaction, and
        input->rule / rule->target edges.
    """
    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    genes = fbn_network["genes"]
    graph = nx.MultiDiGraph()
    for gene in genes:
        graph.add_node(gene, node_kind="gene")

    interactions = fbn_network.get("interactions", {})
    for target_gene in genes:
        interactions_for_gene = interactions.get(target_gene, {})

        if show_decay:
            graph.add_edge(target_gene, target_gene, kind="decay", color=DECAY_COLOR, style="dashed")

        for name, interaction in _interaction_items(interactions_for_gene, target_gene):
            interaction_type = interaction.get("type", 1)
            timestep = interaction.get("timestep", 1)
            expression = interaction.get("expression", "")
            kind = "activator" if interaction_type == 1 else "inhibitor"
            edge_color = ACTIVATOR_COLOR if kind == "activator" else INHIBITOR_COLOR

            rule_node = f"{target_gene}::{name}"
            graph.add_node(
                rule_node,
                node_kind="rule",
                kind=kind,
                timestep=timestep,
                label=f"{'+' if kind == 'activator' else '-'}{timestep}",
                expression=expression,
            )

            tokens = fbnnet_utils.splitExpression(expression, 1, False) if expression else []

            for idx in interaction.get("input", []):
                source_gene = genes[idx - 1]
                negated = source_gene in tokens and tokens.index(source_gene) > 0 and tokens[tokens.index(source_gene) - 1] == "!"
                graph.add_edge(
                    source_gene, rule_node,
                    kind="negated_input" if negated else "input",
                    color=INHIBITOR_COLOR if negated else ACTIVATOR_COLOR,
                    style="solid",
                )

            graph.add_edge(
                rule_node, target_gene,
                kind=kind,
                color=edge_color,
                style="solid" if kind == "activator" else "dashed",
                timestep=timestep,
                probability=interaction.get("probability"),
                expression=expression,
            )

    return graph


def draw_static_network(
    fbn_network: Dict[str, Any],
    ax=None,
    show_decay: bool = False,
    show_rule_nodes: bool = False,
    layout=None,
    figsize=(8, 6),
):
    """
    Draw a static snapshot of the FBN network's regulatory graph.

    Port of R's `StaticNetwork`/`FBNNetwork.Graph(type = "static")`. Activator
    edges are drawn in green, inhibitor edges in red, and (optionally) decay
    self-loops as dashed grey edges.

    Args:
        fbn_network: The Fundamental Boolean Network
        ax: Optional matplotlib Axes to draw on; a new figure/axes is created
            if omitted
        show_decay: If True, draw decay self-loops
        show_rule_nodes: If True, draw the network as
            `gene -> rule(+/-, timestep) -> gene` (one square node per
            activator/inhibitor rule, labelled with its type and timestep,
            with per-input negation highlighting) instead of collapsing each
            rule into a single direct gene->gene edge.
        layout: Optional callable `graph -> pos dict` (defaults to
            `networkx.spring_layout`)
        figsize: Figure size used when `ax` is not provided

    Returns:
        The matplotlib Axes the network was drawn on.
    """
    import matplotlib.pyplot as plt

    if show_rule_nodes:
        graph = to_networkx_graph_with_rules(fbn_network, show_decay=show_decay)
    else:
        graph = to_networkx_graph(fbn_network, show_decay=show_decay)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pos = layout(graph) if layout is not None else nx.spring_layout(graph, seed=42)

    gene_node_size = 800
    rule_node_size = 400
    node_size_by_id = {
        n: (rule_node_size if data.get("node_kind") == "rule" else gene_node_size)
        for n, data in graph.nodes(data=True)
    }
    node_size_array = [node_size_by_id[n] for n in graph.nodes()]

    gene_nodes = [n for n, data in graph.nodes(data=True) if data.get("node_kind", "gene") == "gene"]
    nx.draw_networkx_nodes(graph, pos, nodelist=gene_nodes, ax=ax, node_color="lightblue", node_size=gene_node_size)
    nx.draw_networkx_labels(graph, pos, labels={n: n for n in gene_nodes}, ax=ax, font_size=9)

    if show_rule_nodes:
        for kind, color in (("activator", "lightgreen"), ("inhibitor", "orange")):
            rule_nodes = [n for n, data in graph.nodes(data=True) if data.get("node_kind") == "rule" and data.get("kind") == kind]
            if rule_nodes:
                nx.draw_networkx_nodes(graph, pos, nodelist=rule_nodes, ax=ax, node_color=color, node_shape="s", node_size=rule_node_size)
        rule_labels = {n: data["label"] for n, data in graph.nodes(data=True) if data.get("node_kind") == "rule"}
        nx.draw_networkx_labels(graph, pos, labels=rule_labels, ax=ax, font_size=7)

    for kind, color, style in (
        ("activator", ACTIVATOR_COLOR, "solid"),
        ("inhibitor", INHIBITOR_COLOR, "solid"),
        ("input", ACTIVATOR_COLOR, "solid"),
        ("negated_input", INHIBITOR_COLOR, "solid"),
        ("decay", DECAY_COLOR, "dashed"),
    ):
        edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get("kind") == kind]
        if edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=edges, ax=ax,
                edge_color=color, style=style, arrows=True,
                arrowstyle="-|>", arrowsize=15, node_size=node_size_array,
                connectionstyle="arc3,rad=0.15",
            )

    ax.set_axis_off()
    return ax


def _gene_state_at_timepoint(timeseries_matrix, genes: List[str], time_point: int) -> Dict[str, int]:
    """
    Extract a `gene -> 0/1` state dict for `time_point` (1-based) from a
    genes x timepoints timeseries matrix.

    Accepts either a `pandas.DataFrame` indexed by gene name (columns are
    time point labels, matched by `str(time_point)` if present, otherwise by
    1-based positional column index) or a plain 2-D array/list of lists in
    the same row order as `genes`.
    """
    if isinstance(timeseries_matrix, pd.DataFrame):
        col_label = str(time_point)
        column = timeseries_matrix[col_label] if col_label in timeseries_matrix.columns else timeseries_matrix.iloc[:, time_point - 1]
        return {gene: int(column.loc[gene]) if gene in column.index else int(column.iloc[idx]) for idx, gene in enumerate(genes)}

    matrix = np.asarray(timeseries_matrix)
    return {gene: int(matrix[idx, time_point - 1]) for idx, gene in enumerate(genes)}


def draw_static_network_slice(
    fbn_network: Dict[str, Any],
    timeseries_matrix,
    time_point: int = 1,
    ax=None,
    show_rule_nodes: bool = False,
    layout=None,
    figsize=(8, 6),
):
    """
    Draw the static network with gene nodes colored by their observed
    boolean state at a single timepoint in `timeseries_matrix`.

    Port of R's `FBNNetwork.Graph(type = "staticSlice")` / `StaticNetworkInSlice`.
    R's version additionally restricts edges to only those whose
    probabilistically-simulated activation window (built via its internal
    `convert_to_NGO`) covers `time_point`; that stochastic edge-firing
    simulation has no exact-value ground truth to verify against (R's own
    test only asserts "no error" for this feature), so this pragmatically
    draws the full static graph with gene nodes colored by their actual
    observed state (pink = 0, lightblue = 1) at `time_point` instead.

    Args:
        fbn_network: The Fundamental Boolean Network
        timeseries_matrix: A genes x timepoints `pandas.DataFrame` (indexed
            by gene name) or 2-D array in `fbn_network["genes"]` row order
        time_point: The (1-based) timepoint to color gene nodes by
        ax: Optional matplotlib Axes to draw on
        show_rule_nodes: See `draw_static_network`
        layout: Optional callable `graph -> pos dict`
        figsize: Figure size used when `ax` is not provided

    Returns:
        The matplotlib Axes the network was drawn on.
    """
    import matplotlib.pyplot as plt

    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    genes = fbn_network["genes"]
    states = _gene_state_at_timepoint(timeseries_matrix, genes, time_point)

    graph = to_networkx_graph_with_rules(fbn_network) if show_rule_nodes else to_networkx_graph(fbn_network)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pos = layout(graph) if layout is not None else nx.spring_layout(graph, seed=42)

    gene_nodes = [n for n, data in graph.nodes(data=True) if data.get("node_kind", "gene") == "gene"]
    node_colors = ["lightblue" if states.get(n, 1) else "pink" for n in gene_nodes]
    nx.draw_networkx_nodes(graph, pos, nodelist=gene_nodes, ax=ax, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(graph, pos, labels={n: n for n in gene_nodes}, ax=ax, font_size=9)

    node_size_by_id = {n: (400 if graph.nodes[n].get("node_kind") == "rule" else 800) for n in graph.nodes()}
    node_size_array = [node_size_by_id[n] for n in graph.nodes()]

    if show_rule_nodes:
        for kind, color in (("activator", "lightgreen"), ("inhibitor", "orange")):
            rule_nodes = [n for n, data in graph.nodes(data=True) if data.get("node_kind") == "rule" and data.get("kind") == kind]
            if rule_nodes:
                nx.draw_networkx_nodes(graph, pos, nodelist=rule_nodes, ax=ax, node_color=color, node_shape="s", node_size=400)
        rule_labels = {n: data["label"] for n, data in graph.nodes(data=True) if data.get("node_kind") == "rule"}
        nx.draw_networkx_labels(graph, pos, labels=rule_labels, ax=ax, font_size=7)

    for kind, color, style in (
        ("activator", ACTIVATOR_COLOR, "solid"),
        ("inhibitor", INHIBITOR_COLOR, "solid"),
        ("input", ACTIVATOR_COLOR, "solid"),
        ("negated_input", INHIBITOR_COLOR, "solid"),
    ):
        edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get("kind") == kind]
        if edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=edges, ax=ax,
                edge_color=color, style=style, arrows=True,
                arrowstyle="-|>", arrowsize=15, node_size=node_size_array,
                connectionstyle="arc3,rad=0.15",
            )

    ax.set_axis_off()
    ax.set_title(f"Fundamental Boolean Network at time point {time_point}")
    return ax


def draw_dynamic_network(
    fbn_network: Dict[str, Any],
    timeseries_matrix,
    from_time_point: int = 1,
    to_time_point: int = 5,
    show_rule_nodes: bool = False,
    figsize=None,
):
    """
    Draw a sequence of static-network snapshots across a time range, with
    gene nodes colored by their observed boolean state at each timepoint.

    Pragmatic port of R's `FBNNetwork.Graph(type = "dynamic")` /
    `GenerateDynamicNetworkGraphicObject` (an interactive visNetwork widget
    chaining per-timestep node/edge subsets driven by a probabilistic
    simulation, with no direct static-image equivalent). Here this renders
    one subplot per timepoint in `[from_time_point, to_time_point]`, each
    produced by `draw_static_network_slice` sharing one fixed layout so
    genes stay in the same position across snapshots.

    Args:
        fbn_network: The Fundamental Boolean Network
        timeseries_matrix: A genes x timepoints `pandas.DataFrame` or 2-D
            array, see `draw_static_network_slice`
        from_time_point: First (1-based) timepoint to render
        to_time_point: Last (1-based) timepoint to render (inclusive)
        show_rule_nodes: See `draw_static_network`
        figsize: Figure size for the whole subplot row; defaults to scaling
            with the number of timepoints

    Returns:
        The list of matplotlib Axes (one per timepoint).
    """
    import matplotlib.pyplot as plt

    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")
    if to_time_point < from_time_point:
        raise ValueError("to_time_point must be >= from_time_point")

    time_points = list(range(from_time_point, to_time_point + 1))
    if figsize is None:
        figsize = (6 * len(time_points), 6)

    _, axes = plt.subplots(1, len(time_points), figsize=figsize)
    axes = [axes] if len(time_points) == 1 else list(axes)

    layout_graph = to_networkx_graph_with_rules(fbn_network) if show_rule_nodes else to_networkx_graph(fbn_network)
    shared_layout = nx.spring_layout(layout_graph, seed=42)

    for sub_ax, time_point in zip(axes, time_points):
        draw_static_network_slice(
            fbn_network, timeseries_matrix, time_point=time_point, ax=sub_ax,
            show_rule_nodes=show_rule_nodes, layout=lambda g: shared_layout,
        )

    return axes


def plot_network(
    fbn_network: Dict[str, Any],
    target_genes: Optional[List[str]] = None,
    direction: str = "static",
    regulation_type: Optional[int] = None,
    target_type: Optional[int] = None,
    max_deep: int = 2,
    next_level_mix_type: bool = False,
    output_network: bool = False,
    show_decay: bool = False,
    show_rule_nodes: bool = False,
    timeseries_matrix=None,
    start_time_point: int = 1,
    end_time_point: int = 1,
    target_time_point: int = 1,
    ax=None,
):
    """
    Filter (optionally) and draw an FBN network in one call.

    Port of R's `plotNetwork`, scoped down to the conceptual modes it
    supports (the R version's many `forward_1a`..`backward_2b` variants are
    all just `find_forward_related_network_by_genes`/
    `find_backward_related_network_by_genes` calls with different
    `regulation_type`/`target_type`/`next_level_mix_type` combinations, which
    can be passed directly here).

    Args:
        fbn_network: The Fundamental Boolean Network
        target_genes: Genes to filter/expand around; required for "forward"
            and "backward", optional for "static"/"staticSlice"/"dynamic"
        direction: One of "static", "staticSlice", "dynamic", "forward", "backward"
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        target_type: For "forward" only; 1/0/None, see
            `find_forward_related_network_by_genes`
        max_deep: How many layers of indirection to drill down for "forward"/"backward"
        next_level_mix_type: See `find_forward_related_network_by_genes`/`find_backward_related_network_by_genes`
        output_network: If True, also return the (possibly filtered) network
        show_decay: If True, draw decay self-loops ("static"/"dynamic" only)
        show_rule_nodes: If True, draw as `gene -> rule(+/-, timestep) -> gene` (see `draw_static_network`)
        timeseries_matrix: Required for "staticSlice"/"dynamic"; see `draw_static_network_slice`
        start_time_point: "dynamic" only -- first (1-based) timepoint to render
        end_time_point: "dynamic" only -- last (1-based) timepoint to render (inclusive)
        target_time_point: "staticSlice" only -- the (1-based) timepoint to color gene nodes by
        ax: Optional matplotlib Axes to draw on (ignored for "dynamic", which returns a list of Axes)

    Returns:
        The matplotlib Axes (or list of Axes for "dynamic"), or
        `(network, ax)` if `output_network` is True.
    """
    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    network = fbn_network
    if direction in ("static", "staticSlice", "dynamic"):
        if target_genes:
            network = filter_network_connections_by_genes(network, genelist=target_genes, exclusive=False, expand=False)
    elif direction == "forward":
        if not target_genes:
            raise ValueError("target_genes is required for direction='forward'")
        network = find_forward_related_network_by_genes(
            network, target_gene_list=target_genes,
            regulation_type=regulation_type, target_type=target_type,
            max_deep=max_deep, next_level_mix_type=next_level_mix_type,
        )
    elif direction == "backward":
        if not target_genes:
            raise ValueError("target_genes is required for direction='backward'")
        network = find_backward_related_network_by_genes(
            network, target_gene_list=target_genes,
            regulation_type=regulation_type,
            max_deep=max_deep, next_level_mix_type=next_level_mix_type,
        )
    else:
        raise ValueError(
            f"Unsupported direction '{direction}', expected 'static', 'staticSlice', 'dynamic', 'forward' or 'backward'"
        )

    if direction == "staticSlice":
        if timeseries_matrix is None:
            raise ValueError("timeseries_matrix is required for direction='staticSlice'")
        result_ax = draw_static_network_slice(
            network, timeseries_matrix, time_point=target_time_point, ax=ax, show_rule_nodes=show_rule_nodes,
        )
    elif direction == "dynamic":
        if timeseries_matrix is None:
            raise ValueError("timeseries_matrix is required for direction='dynamic'")
        result_ax = draw_dynamic_network(
            network, timeseries_matrix, from_time_point=start_time_point, to_time_point=end_time_point,
            show_rule_nodes=show_rule_nodes,
        )
    else:
        result_ax = draw_static_network(network, ax=ax, show_decay=show_decay, show_rule_nodes=show_rule_nodes)

    if output_network:
        return network, result_ax
    return result_ax


def draw_attractor(fbm_attractors: Dict[str, Any], index: int = 0, ax=None, figsize=(8, 4)):
    """
    Draw one attractor cycle as a genes x timesteps state heatmap.

    Simplified port of R's `FBNNetwork.Graph.DrawAttractor`, which animates
    the network graph over the attractor's cycle; here we render the
    equivalent information (which genes are on/off at each step of the
    cycle) as a static heatmap instead.

    Args:
        fbm_attractors: The result of `search_for_attractors` (an `FBMAttractors` dict)
        index: Which attractor (0-based) to draw
        ax: Optional matplotlib Axes to draw on
        figsize: Figure size used when `ax` is not provided

    Returns:
        The matplotlib Axes the attractor was drawn on.
    """
    import matplotlib.pyplot as plt

    if fbm_attractors.get("class") != "FBMAttractors":
        raise ValueError("fbm_attractors must be the result of search_for_attractors")

    attractors = fbm_attractors["Attractors"]
    if not (0 <= index < len(attractors)):
        raise ValueError(f"index {index} out of range for {len(attractors)} attractors")

    genes = fbm_attractors["Genes"]
    cycle = attractors[index]
    matrix = [[state[gene] for state in cycle] for gene in genes]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(matrix, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_xticks(range(len(cycle)))
    ax.set_xticklabels([str(i) for i in range(len(cycle))])
    # Annotate each cell with its 0/1 value so all-0/all-1 rows/cycles (e.g.
    # a fixed point) still show readable content instead of a blank image.
    for row, gene in enumerate(genes):
        for col in range(len(cycle)):
            value = matrix[row][col]
            ax.text(col, row, str(value), ha="center", va="center",
                     color="white" if value else "black", fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(len(cycle) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(genes) + 1)], minor=True)
    ax.grid(which="minor", color="lightgrey", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Step in cycle")
    ax.set_title(f"Attractor {index} (length {len(cycle)})")
    return ax
