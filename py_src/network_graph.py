"""
Network visualisation helpers.

A pragmatic Python port of R's `graph_FBN.R` / `plot_network_FBN.R`. Rather than
reproducing the exact visNetwork/igraph JSON structures those files build (which
target an interactive JS widget with no direct Python equivalent), this module
converts an FBN network into a `networkx` graph and renders it with
`matplotlib`, covering the same conceptual operations:

* `to_networkx_graph`      -- port of `ConvertToNetworkGraphicObject`
* `draw_static_network`    -- port of `FBNNetwork.Graph(type = "static")` / `StaticNetwork`
* `plot_network`           -- port of `plotNetwork` (filters, then draws)
* `draw_attractor`         -- port of `FBNNetwork.Graph.DrawAttractor`
"""
import logging
import os
from typing import Any, Dict, List, Optional

import networkx as nx
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
    ax=None,
):
    """
    Filter (optionally) and draw an FBN network in one call.

    Port of R's `plotNetwork`, scoped down to the three conceptual modes it
    supports (the R version's many `forward_1a`..`backward_2b` variants are
    all just `find_forward_related_network_by_genes`/
    `find_backward_related_network_by_genes` calls with different
    `regulation_type`/`target_type`/`next_level_mix_type` combinations, which
    can be passed directly here).

    Args:
        fbn_network: The Fundamental Boolean Network
        target_genes: Genes to filter/expand around; required for "forward"
            and "backward", optional for "static"
        direction: One of "static", "forward", "backward"
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        target_type: For "forward" only; 1/0/None, see
            `find_forward_related_network_by_genes`
        max_deep: How many layers of indirection to drill down for "forward"/"backward"
        next_level_mix_type: See `find_forward_related_network_by_genes`/`find_backward_related_network_by_genes`
        output_network: If True, also return the (possibly filtered) network
        show_decay: If True, draw decay self-loops
        show_rule_nodes: If True, draw as `gene -> rule(+/-, timestep) -> gene` (see `draw_static_network`)
        ax: Optional matplotlib Axes to draw on

    Returns:
        The matplotlib Axes, or `(network, ax)` if `output_network` is True.
    """
    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    network = fbn_network
    if direction == "static":
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
        raise ValueError(f"Unsupported direction '{direction}', expected 'static', 'forward' or 'backward'")

    ax = draw_static_network(network, ax=ax, show_decay=show_decay, show_rule_nodes=show_rule_nodes)

    if output_network:
        return network, ax
    return ax


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
