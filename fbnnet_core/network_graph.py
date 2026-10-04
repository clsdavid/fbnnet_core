"""
Network graphs of Fundamental Boolean Networks.

Port of R's ``graph_FBN.R`` and ``plot_network_FBN.R``. The R package renders its graphs with the
interactive ``visNetwork`` widget; here every graph is an :class:`FBNGraph` that holds the same node and
edge tables (same colours, shapes, line styles, legends and titles) and can be

* drawn with matplotlib (``graph.draw()``, ``graph.save("network.png")``), or
* exported as an interactive web page (``graph.save_html("network.html")``), shown inline in Jupyter.

Main functions
--------------
* ``plot_network``                 -- ``plotNetwork``: filter and draw a network in one call
* ``fbn_network_graph``            -- ``FBNNetwork.Graph`` (static / staticSlice / dynamic)
* ``draw_attractor``               -- ``FBNNetwork.Graph.DrawAttractor``
* ``draw_dynamic_for_one_matrix``  -- ``FBNNetwork.Graph.DrawDynamicForOneMatrix``
* ``convert_to_network_graphic_object`` / ``convert_to_ngo`` -- the underlying graphic objects
* ``to_networkx_graph`` / ``to_networkx_graph_with_rules``   -- ``networkx`` exports
"""
import html
import json
import logging
import random
import re
from functools import lru_cache
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from . import _utils

from .attractor import get_probability_from_function_input
from .fbn_types import _fmt_num
from .network_app import (
    _as_gene_value_dict,
    _interaction_items,
    filter_network_connections_by_genes,
    find_forward_related_network_by_genes,
    find_backward_related_network_by_genes,
)

logger = logging.getLogger(__name__)

_VIS_NETWORK_URL = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>$title</title>
<script src="$url"></script>
<style>
  body { font-family: Helvetica, Arial, sans-serif; margin: 0; }
  h3 { text-align: center; margin: 12px 0; }
  #wrapper { position: relative; }
  #network { width: 100%; height: ${height}px; border: 1px solid lightgray; }
  #legend { position: absolute; top: 8px; right: 8px; background: rgba(255,255,255,.92); border: 1px solid lightgray;
            padding: 6px 10px; font-size: 12px; }
  #legend .item { margin: 3px 0; white-space: nowrap; }
  #legend .swatch { display: inline-block; width: 14px; height: 14px; border: 1px solid #2B7CE9; margin-right: 6px; vertical-align: middle; }
  #legend .line { display: inline-block; width: 26px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>
<h3>$title</h3>
<div id="wrapper"><div id="network"></div><div id="legend">$legend</div></div>
<script>
  var d = $data;
  var network = new vis.Network(document.getElementById("network"),
      {nodes: new vis.DataSet(d.nodes), edges: new vis.DataSet(d.edges)}, d.options);
</script>
</body>
</html>
""")


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

            tokens = _utils.splitExpression(expression, 1, False) if expression else []

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


# ---------------------------------------------------------------------------
# R-faithful graphic objects (port of R/graph_FBN.R)
# ---------------------------------------------------------------------------

_NODE_COLUMNS = ["id", "shape", "color", "type", "value", "label", "shadow", "group", "title"]
_EDGE_COLUMNS = [
    "from", "to", "support", "targetNode", "title", "arrowtail", "arrowhead", "color", "lty",
    "arrow.mode", "arrow.size", "type", "arrows", "label", "dashes", "shadow",
]
_DYNAMIC_COLUMNS = [
    "onset", "terminus", "tail", "head", "onset.censored", "terminus.censored", "duration",
    "edge.id", "fromState", "functype",
]

_NODE_BORDER = "#2B7CE9"  # vis.js default node border colour
_FIGURE_PAD = 1.0  # inches of free space around the drawing

_STATIC_LEGEND_NODES = [
    ("Gene", "ellipse", "lightblue", True),
    ("Activate Function (+, Timestep)", "box", "lightgreen", False),
    ("Inhibit Function (-, Timestep)", "box", "orange", False),
]
_SLICE_LEGEND_NODES = [
    ("Gene", "ellipse", "lightblue", True),
    ("Activate Function (+)", "box", "lightgreen", False),
    ("Inhibit Function (-)", "box", "orange", False),
]
_DYNAMIC_LEGEND_NODES = [
    ("Activated Gene", "ellipse", "lightblue", True),
    ("Inhibited Gene", "ellipse", "pink", True),
    ("Activate Function (+)", "box", "lightgreen", False),
    ("Inhibit Function (-)", "box", "orange", False),
]
_STATIC_LEGEND_EDGES = [
    ("activate", "darkblue", "to", False, True),
    ("inhibit", "darkred", "to", True, True),
    ("activated input", "green", "none", False, False),
    ("deactivated input", "red", "none", True, False),
]
_DYNAMIC_LEGEND_EDGES = [
    ("activate", "darkblue", "to", False, True),
    ("inhibit", "darkred", "to", True, True),
    ("decay", "grey", "to", True, True),
    ("activated input", "green", "none", False, False),
    ("deactivated input", "red", "none", True, False),
]


def _check_network(fbn_network: Dict[str, Any]) -> None:
    if fbn_network.get("class") not in _FBN_NETWORK_CLASSES:
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")


@lru_cache(maxsize=1)
def _bundled_david_titles() -> Dict[str, str]:
    from .datasets import load_dataset

    return _david_titles(load_dataset("DAVID_Gene_List"))


def _david_titles(david_gene_list: pd.DataFrame) -> Dict[str, str]:
    """Gene symbol -> first annotated gene name (what R shows in the node tooltip)."""
    first = david_gene_list.drop_duplicates("Symbol").dropna(subset=["Gene.Name"])
    return dict(zip(first["Symbol"], first["Gene.Name"]))


def _legend_frames(nodes, edges):
    legend_nodes = pd.DataFrame(nodes, columns=["label", "shape", "color", "shadow"])
    legend_edges = pd.DataFrame(edges, columns=["label", "color", "arrows", "dashes", "shadow"])
    return legend_nodes, legend_edges


class FBNGraph:
    """
    A drawable Fundamental Boolean Network graph (the Python counterpart of the
    interactive ``visNetwork`` widgets produced by the R package).

    The graph is described by two data frames, ``nodes`` and ``edges``, with the same
    columns and values as the R graphic objects, plus a title and a legend. It can be drawn
    with matplotlib (:meth:`draw`, :meth:`save`) or exported as a self-contained interactive
    web page (:meth:`to_html`, :meth:`save_html`). In a Jupyter notebook the interactive page
    is shown automatically.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        title: str = "",
        legend_nodes: Optional[pd.DataFrame] = None,
        legend_edges: Optional[pd.DataFrame] = None,
        hierarchical: bool = False,
        levels: Optional[Dict[str, int]] = None,
    ):
        self.nodes = nodes.reset_index(drop=True)
        self.edges = edges
        self.title = title
        self.legend_nodes = legend_nodes if legend_nodes is not None else pd.DataFrame(columns=["label", "shape", "color", "shadow"])
        self.legend_edges = legend_edges if legend_edges is not None else pd.DataFrame(columns=["label", "color", "arrows", "dashes", "shadow"])
        self.hierarchical = hierarchical
        self.levels = levels

    def __repr__(self) -> str:
        return f"<FBNGraph '{self.title}': {len(self.nodes)} nodes, {len(self.edges)} edges>"

    # -- conversion ---------------------------------------------------------
    def to_networkx(self) -> "nx.MultiDiGraph":
        """Return the graph as a ``networkx.MultiDiGraph`` (node/edge columns become attributes)."""
        graph = nx.MultiDiGraph(title=self.title)
        for node in self.nodes.to_dict("records"):
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in self.edges.to_dict("records"):
            graph.add_edge(edge["from"], edge["to"], **{k: v for k, v in edge.items() if k not in ("from", "to")})
        return graph

    # -- layout -------------------------------------------------------------
    def _node_levels(self) -> Optional[Dict[str, int]]:
        if self.levels is not None:
            return self.levels
        if "level" in self.nodes.columns:
            return dict(zip(self.nodes["id"], self.nodes["level"].astype(int)))
        return None

    def _layered_positions(self, levels: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
        """Left-to-right layers; nodes inside a layer are ordered to reduce edge crossings."""
        ids = list(self.nodes["id"])
        by_level: Dict[int, List[str]] = {}
        for node_id in ids:
            by_level.setdefault(levels.get(node_id, 0), []).append(node_id)
        order = sorted(by_level)
        neighbours: Dict[str, List[str]] = {node_id: [] for node_id in ids}
        for source, target in zip(self.edges["from"], self.edges["to"]):
            if source in neighbours and target in neighbours:
                neighbours[source].append(target)
                neighbours[target].append(source)

        title_of = dict(zip(self.nodes["id"], self.nodes["title"].astype(str)))
        slot = {}
        for level in order:
            for rank, node_id in enumerate(sorted(by_level[level], key=lambda n: title_of[n])):
                slot[node_id] = float(rank)
        for _ in range(4):
            for level in order[1:] + order[-2::-1]:
                def barycentre(node_id):
                    near = [slot[n] for n in neighbours[node_id] if levels.get(n, 0) != level and n in slot]
                    return sum(near) / len(near) if near else slot[node_id]

                ranked = sorted(by_level[level], key=barycentre)
                for rank, node_id in enumerate(ranked):
                    slot[node_id] = float(rank)

        positions = {}
        for level_rank, level in enumerate(order):
            count = len(by_level[level])
            for node_id in by_level[level]:
                positions[node_id] = (2.4 * level_rank, -0.6 * (slot[node_id] - (count - 1) / 2.0))
        return positions

    def _physics_positions(self, seed: int) -> Dict[str, Tuple[float, float]]:
        """Force-directed layout per connected component (positions in inches); components are packed in rows."""
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes["id"])
        graph.add_edges_from(zip(self.edges["from"], self.edges["to"]))

        blocks = []
        for component in sorted(nx.connected_components(graph), key=lambda c: (-len(c), sorted(c)[0])):
            sub = graph.subgraph(sorted(component))
            n = len(sub)
            if n == 1:
                pos = {node: (0.0, 0.0) for node in sub}
            else:
                radius = 0.75 * np.sqrt(n) + 0.7
                pos = nx.spring_layout(sub, iterations=300, seed=seed, scale=radius)
            xs = [float(x) for x, _ in pos.values()]
            ys = [float(y) for _, y in pos.values()]
            blocks.append((pos, min(xs), max(xs), min(ys), max(ys)))

        total_area = sum((b[2] - b[1] + 1.6) * (b[4] - b[3] + 1.2) for b in blocks)
        row_width = max(blocks[0][2] - blocks[0][1] + 1.6, 1.4 * np.sqrt(total_area))
        positions: Dict[str, Tuple[float, float]] = {}
        cursor_x = cursor_y = row_height = 0.0
        for pos, x0, x1, y0, y1 in blocks:
            width, height = x1 - x0 + 1.6, y1 - y0 + 1.2
            if cursor_x > 0 and cursor_x + width > row_width:
                cursor_x, cursor_y, row_height = 0.0, cursor_y - row_height, 0.0
            for node, (x, y) in pos.items():
                positions[node] = (cursor_x + 0.8 + float(x) - x0, cursor_y - 0.6 - (y1 - float(y)))
            cursor_x += width
            row_height = max(row_height, height)
        return positions

    def _default_figsize(self, positions) -> Tuple[float, float]:
        xs = [p[0] for p in positions.values()] or [0.0]
        ys = [p[1] for p in positions.values()] or [0.0]
        width = max(xs) - min(xs) + 2 * _FIGURE_PAD
        height = max(ys) - min(ys) + 2 * _FIGURE_PAD + 0.6
        return (min(40.0, max(7.0, width)), min(40.0, max(4.0, height)))

    # -- matplotlib ---------------------------------------------------------
    def draw(self, ax=None, figsize=None, seed: int = 42, show_legend: bool = True, font_size: float = 9.0):
        """
        Draw the graph with matplotlib using the colours, shapes, line styles, legend and title of
        the R widget. Returns the matplotlib ``Axes``.
        """
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Arc, FancyArrowPatch

        levels = self._node_levels()
        positions = self._layered_positions(levels) if levels is not None else self._physics_positions(seed)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize or self._default_figsize(positions))
        fig = ax.figure
        ax.set_axis_off()
        if self.title:
            ax.set_title(self.title, fontsize=font_size + 3, fontweight="bold")

        xs = [p[0] for p in positions.values()] or [0.0]
        ys = [p[1] for p in positions.values()] or [0.0]
        ax.set_xlim(min(xs) - _FIGURE_PAD, max(xs) + _FIGURE_PAD)
        ax.set_ylim(min(ys) - _FIGURE_PAD, max(ys) + _FIGURE_PAD)

        shadow = [pe.withSimplePatchShadow(offset=(2, -2), alpha=0.3)]
        texts = {}
        for node in self.nodes.itertuples(index=False):
            x, y = positions[node.id]
            is_gene = node.shape == "ellipse"
            texts[node.id] = ax.text(
                x, y, node.label, ha="center", va="center",
                fontsize=font_size if is_gene else font_size - 1, zorder=3,
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=1.0" if is_gene else "square,pad=0.35",
                          facecolor=node.color, edgecolor=_NODE_BORDER, linewidth=1.2),
            )
            if bool(node.shadow):
                texts[node.id].get_bbox_patch().set_path_effects(shadow)

        # the arrows are clipped against the node boxes, which need a first draw to get their size
        fig.canvas.draw()

        pairs = set(zip(self.edges["from"], self.edges["to"]))
        for edge in self.edges.itertuples(index=False):
            source, target = edge[self.edges.columns.get_loc("from")], edge[self.edges.columns.get_loc("to")]
            color = edge.color
            linestyle = (0, (6, 4)) if bool(edge.dashes) else "solid"
            line_effects = [pe.SimpleLineShadow(offset=(1.5, -1.5), alpha=0.25), pe.Normal()] if bool(edge.shadow) else None
            if source == target:
                x, y = positions[source]
                box = texts[source].get_window_extent().transformed(ax.transData.inverted())
                width = max(box.width * 0.5, 0.05)
                loop = Arc((x, box.y1 + 0.15 * box.height), width, box.height * 1.1, theta1=-30, theta2=210,
                           color=color, linestyle=linestyle, linewidth=1.4, zorder=2)
                if line_effects:
                    loop.set_path_effects(line_effects)
                ax.add_patch(loop)
                continue
            rad = 0.18 if (target, source) in pairs else 0.0

            def arrow_patch(arrowstyle, linewidth, style, zorder=2):
                return FancyArrowPatch(
                    positions[source], positions[target], arrowstyle=arrowstyle, mutation_scale=14,
                    patchA=texts[source].get_bbox_patch(), patchB=texts[target].get_bbox_patch(),
                    shrinkA=1, shrinkB=1, color=color, linewidth=linewidth, linestyle=style,
                    connectionstyle=f"arc3,rad={rad}", zorder=zorder,
                )

            has_head = edge.arrows == "to"
            patch = arrow_patch("-", 1.4, linestyle)
            if line_effects:
                patch.set_path_effects(line_effects)
            ax.add_patch(patch)
            if has_head:
                # a dashed line style would also dash the arrow head, so the head is drawn separately
                ax.add_patch(arrow_patch("-|>,head_length=0.6,head_width=0.25", 0.5, "solid", zorder=4))

        if show_legend and (len(self.legend_nodes) or len(self.legend_edges)):
            handles = []
            for row in self.legend_nodes.itertuples(index=False):
                handles.append(Line2D([], [], linestyle="none", marker="o" if row.shape == "ellipse" else "s",
                                      markersize=11, markerfacecolor=row.color, markeredgecolor=_NODE_BORDER, label=row.label))
            for row in self.legend_edges.itertuples(index=False):
                handles.append(Line2D([], [], color=row.color, linewidth=2, linestyle=(0, (4, 2)) if bool(row.dashes) else "solid",
                                      label=row.label))
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=font_size - 1,
                      title="Legend", title_fontsize=font_size)
        return ax

    def save(self, path: str, dpi: int = 200, **kwargs) -> str:
        """Save the graph; ``.html`` writes the interactive page, any other extension (png, pdf, svg) a matplotlib figure."""
        if str(path).lower().endswith((".html", ".htm")):
            return self.save_html(path, **kwargs)
        import matplotlib.pyplot as plt

        ax = self.draw(**kwargs)
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(ax.figure)
        return str(path)

    def show(self, **kwargs):
        """Draw the graph and open a matplotlib window (needs an interactive matplotlib backend)."""
        import matplotlib.pyplot as plt

        self.draw(**kwargs)
        plt.show()

    # -- interactive HTML ---------------------------------------------------
    def to_html(self, height: int = 700, vis_network_url: str = _VIS_NETWORK_URL) -> str:
        """Return a self-contained web page with an interactive (draggable, zoomable) vis-network graph."""
        nodes = json.loads(self.nodes.to_json(orient="records"))
        for node in nodes:
            node["title"] = html.escape(str(node.get("title", "")))
        edges = json.loads(self.edges.to_json(orient="records"))
        edges = [
            {"from": e["from"], "to": e["to"], "arrows": e["arrows"], "dashes": bool(e["dashes"]), "color": e["color"],
             "shadow": bool(e["shadow"]), "title": html.escape(str(e["title"])), "width": 1.5}
            for e in edges
        ]
        options: Dict[str, Any] = {
            "interaction": {"dragNodes": True, "dragView": True, "zoomView": True},
            "nodes": {"font": {"size": 14}},
        }
        if self.hierarchical:
            options["layout"] = {"hierarchical": {"enabled": True, "direction": "LR", "levelSeparation": 200}}

        legend = []
        for row in self.legend_nodes.itertuples(index=False):
            radius = "50%" if row.shape == "ellipse" else "2px"
            legend.append(
                f'<div class="item"><span class="swatch" style="background:{html.escape(row.color)};border-radius:{radius}"></span>'
                f"{html.escape(row.label)}</div>"
            )
        for row in self.legend_edges.itertuples(index=False):
            style = "dashed" if bool(row.dashes) else "solid"
            legend.append(
                f'<div class="item"><span class="line" style="border-top:3px {style} {html.escape(row.color)}"></span>'
                f"{html.escape(row.label)}</div>"
            )
        data = {"nodes": nodes, "edges": edges, "options": options}
        return _HTML_TEMPLATE.substitute(
            title=html.escape(self.title or "Fundamental Boolean Network"),
            height=int(height),
            url=html.escape(vis_network_url),
            legend="".join(legend),
            data=json.dumps(data).replace("</", "<\\/"),
        )

    def save_html(self, path: str, **kwargs) -> str:
        """Write the interactive page to ``path`` (needs internet access to load vis-network when opened)."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_html(**kwargs))
        return str(path)

    def _repr_html_(self) -> str:  # Jupyter
        return f'<iframe srcdoc="{html.escape(self.to_html(height=600), quote=True)}" width="100%" height="640" frameborder="0"></iframe>'


def convert_to_network_graphic_object(
    fbn_network: Dict[str, Any],
    david_gene_list: Optional[pd.DataFrame] = None,
    show_decay: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Convert an FBN network into the node/edge tables used for drawing (R's ``ConvertToNetworkGraphicObject``).

    Every gene becomes a light-blue ellipse and every activator/inhibitor rule a box (light green
    ``+, timestep`` / orange ``-, timestep``). Rules are connected ``input gene -> rule -> target gene``;
    inputs are green when the gene must be on and dashed red when it must be off.

    Args:
        fbn_network: The Fundamental Boolean Network
        david_gene_list: Optional DAVID annotation (columns ``Symbol`` and ``Gene.Name``) used for the gene tooltips
        show_decay: If True, add the decay self-loop of every gene

    Returns:
        ``{"nodes": DataFrame, "edges": DataFrame}``
    """
    _check_network(fbn_network)
    genes = list(fbn_network["genes"])
    titles = _david_titles(david_gene_list) if david_gene_list is not None else None

    nodes = [
        dict(id=g, shape="ellipse", color="lightblue", type="gene", value=4, label=g, shadow=True, group="Gene",
             title=(titles.get(g, g) if titles is not None else g))
        for g in genes
    ]
    edges: List[Dict[str, Any]] = []

    def add_edge(**kwargs):
        edges.append(kwargs)

    interactions = fbn_network.get("interactions", {})
    for target in interactions:
        if show_decay:
            add_edge(**{"from": target, "to": target, "support": 0, "targetNode": target, "title": target, "arrowtail": "none",
                        "arrowhead": "odot", "color": "grey", "lty": "longdash", "arrow.mode": 2, "arrow.size": 0.2,
                        "type": "decay", "arrows": "to", "label": "", "dashes": True, "shadow": True})
        for rule_name, rule in _interaction_items(interactions[target], target):
            inhibitor = float(rule.get("type", 1)) == 0
            timestep = _fmt_num(rule.get("timestep", 1) if rule.get("timestep") is not None else 1)
            expression = rule.get("expression", "")
            support = float(rule["support"]) if rule.get("support") is not None else float("nan")
            nodes.append(dict(
                id=rule_name, shape="box", color="orange" if inhibitor else "LightGreen", type="TF", value=2,
                label=f"{'-' if inhibitor else '+'}, {timestep}", shadow=False,
                group="Inhibit Function" if inhibitor else "Activate Function", title=expression,
            ))
            add_edge(**{"from": rule_name, "to": target, "support": support, "targetNode": target, "title": rule_name,
                        "arrowtail": "none", "arrowhead": "odot" if inhibitor else "open",
                        "color": "darkred" if inhibitor else "darkblue", "lty": "longdash" if inhibitor else "solid",
                        "arrow.mode": 2, "arrow.size": 0.2, "type": "TF_to_Gene", "arrows": "to", "label": "",
                        "dashes": inhibitor, "shadow": True})
            tokens = _utils.splitExpression(expression, 1, False)
            for index in rule.get("input", []):
                gene = genes[int(index) - 1]
                if gene not in tokens:
                    raise ValueError("The input gene index should not be zero")
                position = tokens.index(gene)
                negated = position >= 1 and tokens[position - 1] == "!"
                add_edge(**{"from": gene, "to": rule_name, "support": support, "targetNode": target, "title": rule_name,
                            "arrowtail": "tee" if negated else "none", "arrowhead": "none",
                            "color": "red" if negated else "green", "lty": "longdash" if inhibitor else "solid",
                            "arrow.mode": 0, "arrow.size": 0.2, "type": "Gene_to_TF", "arrows": "none", "label": "",
                            "dashes": negated, "shadow": False})

    edge_frame = pd.DataFrame(edges, columns=_EDGE_COLUMNS)
    edge_frame.index = range(1, len(edge_frame) + 1)
    return {"nodes": pd.DataFrame(nodes, columns=_NODE_COLUMNS), "edges": edge_frame}


def _as_timeseries_frame(timeseries) -> pd.DataFrame:
    if timeseries is None or not isinstance(timeseries, pd.DataFrame):
        raise ValueError("The parameter timeseries must be not None and be a DataFrame (genes x time points)")
    return timeseries


def convert_to_ngo(
    timeseries: pd.DataFrame,
    fbn_network: Dict[str, Any],
    network_object: Dict[str, pd.DataFrame],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Work out which regulatory rules (and decays) were active between consecutive time points
    (R's ``convert_to_NGO``).

    Args:
        timeseries: Binary genes x time points ``DataFrame`` (row names are the genes)
        fbn_network: The Fundamental Boolean Network
        network_object: The result of :func:`convert_to_network_graphic_object` (with ``show_decay=True``
            if decay edges are wanted)
        seed: Seed of the random draw that decides whether a rule with a probability below 1 fires

    Returns:
        A ``DataFrame`` with columns ``onset, terminus, tail, head, onset.censored, terminus.censored, duration,
        edge.id, fromState, functype`` (one row per interval in which an edge was active).
    """
    timeseries = _as_timeseries_frame(timeseries)
    _check_network(fbn_network)
    if network_object is None:
        raise ValueError("The networkobject is required")

    rng = random.Random(seed)
    genes = [str(g) for g in timeseries.index]
    values = timeseries.to_numpy()
    n_steps = values.shape[1]
    interactions = fbn_network.get("interactions", {})
    timedecay = _as_gene_value_dict(fbn_network.get("timedecay", {}), list(fbn_network["genes"]), default=1)

    nodes, edges = network_object["nodes"], network_object["edges"]
    node_position = {}
    for position, node_id in enumerate(nodes["id"], start=1):
        node_position.setdefault(node_id, position)
    rule_edges: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = {}
    decay_edges: Dict[str, List[Tuple[int, str, str]]] = {}
    for edge_id, source, target, title, target_node, edge_type in zip(
        edges.index, edges["from"], edges["to"], edges["title"], edges["targetNode"], edges["type"]
    ):
        rule_edges.setdefault((target_node, title), []).append((edge_id, source, target))
        if edge_type == "decay":
            decay_edges.setdefault(target_node, []).append((edge_id, source, target))

    records: Dict[int, List[tuple]] = {}

    def record(edge_list, step, selected_name, function_input, kind):
        for edge_id, source, target in edge_list:
            state = function_input.get(source)
            records.setdefault(edge_id, []).append(
                (step - 1, step, node_position[source], node_position[target],
                 float(state) if state is not None else float("nan"), selected_name, kind)
            )

    for step in range(2, n_steps + 1):
        previous = values[:, step - 2]
        decay_index = {gene: 1 for gene in genes}
        for gene_position, gene in enumerate(genes):
            rules = [(name, rule) for name, rule in _interaction_items(interactions.get(gene, {}), gene)]
            decay = timedecay.get(gene, 1)

            def fired(rule_type):
                chosen = []
                for name, rule in rules:
                    if float(rule.get("type", 1)) != rule_type:
                        continue
                    pre_input = {genes[int(i) - 1]: previous[int(i) - 1] for i in rule.get("input", [])}
                    probability = rule.get("probability")
                    p = get_probability_from_function_input(
                        int(rule_type), rule.get("expression", ""), 1.0 if probability is None else probability, pre_input)
                    if rng.random() < p:
                        chosen.append((name, pre_input))
                return chosen

            activations = fired(1.0)
            inhibitions = fired(0.0)

            decays = []
            if decay is not None and decay > 0:
                if activations or inhibitions:
                    decay_index[gene] = 1
                elif decay_index[gene] >= decay:
                    decay_index[gene] = 1
                    decays.append(("decay", {gene: previous[gene_position] == 1}))
                else:
                    decay_index[gene] += 1

            if inhibitions:
                activations = []
            for name, function_input in activations:
                record(rule_edges.get((gene, name), []), step, name, function_input, "activate")
            for name, function_input in inhibitions:
                record(rule_edges.get((gene, name), []), step, name, function_input, "inhibit")
            for name, function_input in decays:
                record(decay_edges.get(gene, []), step, name, function_input, "decay")

    rows = []
    for edge_id, steps in records.items():
        first = steps[0]
        if len(steps) == 1:
            rows.append((first, edge_id))
            continue
        # Same interval-merging rule as the R implementation: runs of consecutive time steps are united, and a
        # trailing isolated step (one that cannot be united with the previous one) is not reported.
        for index in range(1, len(steps)):
            following = steps[index]
            united = following[0] == first[1]
            if united:
                first = (first[0], following[1]) + first[2:]
            else:
                rows.append((first, edge_id))
                first = following
            if index == len(steps) - 1 and united:
                rows.append((first, edge_id))

    return pd.DataFrame(
        [
            (r[0], r[1], r[2], r[3], False, False, r[1] - r[0], edge_id, r[4], r[6])
            for r, edge_id in rows
        ],
        columns=_DYNAMIC_COLUMNS,
    )


def _active_edge_rows(network_object, dynamic_object, timepoint):
    edges = network_object["edges"]
    if timepoint > 0:
        selected = dynamic_object[(dynamic_object["onset"] < timepoint) & (timepoint <= dynamic_object["terminus"])]
    else:
        selected = dynamic_object
    return edges.loc[edges.index.isin(selected["edge.id"])]


def static_network(network_object: Dict[str, pd.DataFrame]) -> FBNGraph:
    """Build the static graph of a graphic object (R's ``StaticNetwork``)."""
    if network_object is None:
        raise ValueError("The networkobject is required")
    legend_nodes, legend_edges = _legend_frames(_STATIC_LEGEND_NODES, _STATIC_LEGEND_EDGES)
    return FBNGraph(network_object["nodes"], network_object["edges"], "Fundamental Boolean Networks", legend_nodes, legend_edges)


def static_network_in_slice(
    network_object: Dict[str, pd.DataFrame], timepoint: int, dynamic_object: pd.DataFrame
) -> FBNGraph:
    """Build the graph of the rules that were active at one time point (R's ``StaticNetworkInSlice``)."""
    if dynamic_object is None:
        raise ValueError("The dynamicNetworkGraphicObject is required")
    if network_object is None:
        raise ValueError("The networkobject is required")

    new_edges = _active_edge_rows(network_object, dynamic_object, timepoint)
    nodes = network_object["nodes"]
    genes = nodes[nodes["type"] == "gene"]
    linked = nodes[nodes["id"].isin(new_edges["from"]) | nodes["id"].isin(new_edges["to"])]
    new_nodes = pd.concat([genes, linked]).drop_duplicates()
    legend_nodes, legend_edges = _legend_frames(_SLICE_LEGEND_NODES, _STATIC_LEGEND_EDGES)
    return FBNGraph(new_nodes, new_edges, f"Fundamental Boolean Networks in the time step of  {timepoint}", legend_nodes, legend_edges)


def _recolour(frame: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    values = frame["id"].map(state)
    frame.loc[values == 0, "color"] = "pink"
    frame.loc[values == 1, "color"] = "lightblue"
    return frame


def _build_step(network_object, dynamic_object, index, current_state, next_state):
    """One time step of a dynamic graph: the active rules between time ``index`` and ``index + 1``."""
    nodes = network_object["nodes"]
    next_index = index + 1
    new_edges = _active_edge_rows(network_object, dynamic_object, next_index).copy()

    from_nodes = nodes[nodes["id"].isin(new_edges["from"])]
    to_nodes = nodes[nodes["id"].isin(new_edges["to"])]
    from_gene = _recolour(from_nodes[from_nodes["type"] == "gene"], current_state)
    from_tf = from_nodes[from_nodes["type"] == "TF"].copy()
    to_gene = _recolour(to_nodes[to_nodes["type"] == "gene"], next_state)
    to_tf = to_nodes[to_nodes["type"] == "TF"].copy()

    from_gene_ids, from_tf_ids = set(from_gene["id"]), set(from_tf["id"])
    to_gene_ids, to_tf_ids = set(to_gene["id"]), set(to_tf["id"])
    new_edges["from"] = [
        f"{n}_{index}" if n in from_gene_ids else (f"{n}_{next_index}" if n in from_tf_ids else n) for n in new_edges["from"]
    ]
    new_edges["to"] = [
        f"{n}_{next_index}" if n in to_gene_ids else (f"{n}_{next_index}" if n in to_tf_ids else n) for n in new_edges["to"]
    ]

    for frame, suffix in ((from_gene, index), (to_gene, next_index)):
        frame["id"] = frame["id"] + f"_{suffix}"
        frame["label"] = frame["label"] + f"_{suffix}"
    from_tf["id"] = from_tf["id"] + f"_{next_index}"
    to_tf["id"] = to_tf["id"] + f"_{next_index}"
    return from_gene, from_tf, to_gene, to_tf, new_edges


def generate_dynamic_network_graphic_object(
    network_object: Dict[str, pd.DataFrame],
    from_time_point: int,
    to_time_point: int,
    dynamic_object: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> FBNGraph:
    """
    Build the dynamic graph: the rules active between consecutive time points, with genes coloured light blue
    (on) or pink (off) at every time point (R's ``GenerateDynamicNetworkGraphicObject``).
    """
    timeseries = _as_timeseries_frame(timeseries)
    if dynamic_object is None:
        raise ValueError("The dynamicNetworkGraphicObject is required")
    if network_object is None:
        raise ValueError("The networkobject is required")
    if not (1 <= from_time_point <= timeseries.shape[1] and to_time_point <= timeseries.shape[1]):
        raise ValueError(f"The time points must be within 1 and {timeseries.shape[1]}")

    pieces_nodes, pieces_edges, levels = [], [], {}
    for index in range(int(from_time_point), int(to_time_point)):
        current = pd.Series(timeseries.iloc[:, index - 1].to_numpy(), index=[str(g) for g in timeseries.index])
        following = pd.Series(timeseries.iloc[:, index].to_numpy(), index=[str(g) for g in timeseries.index])
        from_gene, from_tf, to_gene, to_tf, new_edges = _build_step(network_object, dynamic_object, index, current, following)
        pieces_nodes += [to_gene, from_gene, from_tf, to_tf]
        pieces_edges.append(new_edges)
        for frame, level in ((from_gene, 2 * index - 1), (from_tf, 2 * index), (to_tf, 2 * index), (to_gene, 2 * index + 1)):
            levels.update({node_id: level for node_id in frame["id"]})

    nodes = pd.concat(pieces_nodes).drop_duplicates() if pieces_nodes else pd.DataFrame(columns=_NODE_COLUMNS)
    edges = pd.concat(pieces_edges).drop_duplicates() if pieces_edges else pd.DataFrame(columns=_EDGE_COLUMNS)
    legend_nodes, legend_edges = _legend_frames(_DYNAMIC_LEGEND_NODES, _DYNAMIC_LEGEND_EDGES)
    return FBNGraph(
        nodes, edges,
        f"Dynamic Fundamental Boolean Networks from the time point of {from_time_point} to {to_time_point}",
        legend_nodes, legend_edges, levels=levels,
    )


def _independent_nodes(genes, involved, matrix, column, level, suffix):
    rows = []
    for gene in genes:
        if gene in involved:
            continue
        state = matrix.loc[gene, column]
        rows.append(dict(id=f"{gene}_{suffix}", shape="ellipse", color="pink" if state == 0 else "lightblue", type="gene",
                         value=4, label=f"{gene}_{suffix}", shadow=True, group="Gene", title=gene, level=level))
    return pd.DataFrame(rows, columns=_NODE_COLUMNS + ["level"])


def drawing_attractor_internal(
    network_object: Dict[str, pd.DataFrame], dynamic_object: pd.DataFrame, matrix: pd.DataFrame
) -> FBNGraph:
    """
    Build the layered (left to right) graph of a state sequence such as an attractor cycle, one layer triple
    (genes, rules, genes) per transition (R's ``DrawingAttractorInternal``). Genes that take no part in a
    transition are added unconnected.
    """
    matrix = _as_timeseries_frame(matrix)
    if dynamic_object is None:
        raise ValueError("The dynamicNetworkGraphicObject is required")
    if network_object is None:
        raise ValueError("The networkobject is required")

    genes = [str(g) for g in matrix.index]
    columns = list(matrix.columns)
    pieces_nodes, pieces_edges = [], []
    start_level = 1
    for index in range(1, len(columns)):
        current = pd.Series(matrix.iloc[:, index - 1].to_numpy(), index=genes)
        following = pd.Series(matrix.iloc[:, index].to_numpy(), index=genes)
        from_gene, from_tf, to_gene, to_tf, new_edges = _build_step(network_object, dynamic_object, index, current, following)
        from_gene = from_gene.assign(level=start_level)
        from_tf = from_tf.assign(level=start_level + 1)
        to_gene = to_gene.assign(level=start_level + 2)
        to_tf = to_tf.assign(level=start_level + 1)
        independent_from = _independent_nodes(genes, set(from_gene["title"]), matrix, columns[index - 1], start_level, index)
        independent_to = _independent_nodes(genes, set(to_gene["title"]), matrix, columns[index], start_level + 2, index + 1)
        pieces_nodes += [independent_from, independent_to, to_gene, from_gene, from_tf, to_tf]
        pieces_edges.append(new_edges)
        start_level += 2

    nodes = pd.concat([p for p in pieces_nodes if len(p)]).drop_duplicates() if pieces_nodes else pd.DataFrame(columns=_NODE_COLUMNS + ["level"])
    edges = pd.concat(pieces_edges).drop_duplicates() if pieces_edges else pd.DataFrame(columns=_EDGE_COLUMNS)
    legend_nodes, legend_edges = _legend_frames(_DYNAMIC_LEGEND_NODES, _DYNAMIC_LEGEND_EDGES)
    return FBNGraph(
        nodes, edges, f"Dynamic Fundamental Boolean Networks from the time point of 1 to {len(columns)}",
        legend_nodes, legend_edges, hierarchical=True,
    )


def fbn_network_graph(
    fbn_network: Dict[str, Any],
    type: str = "static",
    timeseries_matrix: Optional[pd.DataFrame] = None,
    from_time_point: int = 1,
    to_time_point: int = 5,
    network_object: Optional[Dict[str, pd.DataFrame]] = None,
    david_gene_list: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
) -> Optional[FBNGraph]:
    """
    Create the graph of a Fundamental Boolean Network (R's ``FBNNetwork.Graph``).

    Args:
        fbn_network: The Fundamental Boolean Network
        type: ``"static"`` (all rules), ``"staticSlice"`` (rules active at ``to_time_point``) or ``"dynamic"``
            (rules active between ``from_time_point`` and ``to_time_point``)
        timeseries_matrix: Binary genes x time points ``DataFrame``; required for ``staticSlice`` and ``dynamic``
        from_time_point: First time point of a dynamic graph
        to_time_point: Last time point (the slice time point for ``staticSlice``)
        network_object: A ready made graphic object (see :func:`convert_to_network_graphic_object`)
        david_gene_list: DAVID annotation used for the gene tooltips; the bundled annotation is used by default
        seed: Seed for rules with a probability below 1

    Returns:
        An :class:`FBNGraph`, or ``None`` if the network has no interactions.
    """
    _check_network(fbn_network)
    if len(fbn_network.get("interactions", {})) == 0:
        return None
    if type not in ("static", "staticSlice", "dynamic"):
        raise ValueError('type must be "static", "staticSlice" or "dynamic"')

    if network_object is None:
        if david_gene_list is None:
            from .datasets import load_dataset

            david_gene_list = load_dataset("DAVID_Gene_List")
        network_object = convert_to_network_graphic_object(fbn_network, david_gene_list, show_decay=(type == "dynamic"))

    dynamic_object = None
    if timeseries_matrix is not None:
        if not isinstance(timeseries_matrix, pd.DataFrame):
            raise ValueError("The parameter timeseries_matrix should be a DataFrame (genes x time points)")
        dynamic_object = convert_to_ngo(timeseries_matrix, fbn_network, network_object, seed=seed)

    if type == "static":
        return static_network(network_object)
    if timeseries_matrix is None:
        raise ValueError(f'timeseries_matrix is required for type "{type}"')
    if type == "staticSlice":
        return static_network_in_slice(network_object, to_time_point, dynamic_object)
    return generate_dynamic_network_graphic_object(network_object, from_time_point, to_time_point, dynamic_object, timeseries_matrix)


def draw_dynamic_for_one_matrix(fbn_network: Dict[str, Any], matrix_data: pd.DataFrame, seed: Optional[int] = None) -> FBNGraph:
    """Layered graph of the rules that explain a state sequence (R's ``FBNNetwork.Graph.DrawDynamicForOneMatrix``)."""
    network_object = convert_to_network_graphic_object(fbn_network, show_decay=False)
    dynamic_object = convert_to_ngo(matrix_data, fbn_network, network_object, seed=seed)
    return drawing_attractor_internal(network_object, dynamic_object, matrix_data)


def attractor_to_matrix(fbm_attractors: Dict[str, Any], index: int = 0) -> pd.DataFrame:
    """The states of one attractor cycle as a genes x steps ``DataFrame`` (columns ``1..length``)."""
    if fbm_attractors.get("class") != "FBMAttractors":
        raise ValueError("fbm_attractors must be the result of search_for_attractors")
    attractors = fbm_attractors["Attractors"]
    if not (0 <= index < len(attractors)):
        raise ValueError(f"index {index} out of range for {len(attractors)} attractors")
    genes = fbm_attractors["Genes"]
    cycle = attractors[index]
    return pd.DataFrame([[state[g] for state in cycle] for g in genes], index=genes, columns=range(1, len(cycle) + 1))


def draw_attractor(fbn_network: Dict[str, Any], fbm_attractors: Dict[str, Any], index: int = 0, seed: Optional[int] = None) -> FBNGraph:
    """
    Draw one attractor as a dynamic network (R's ``FBNNetwork.Graph.DrawAttractor``).

    Args:
        fbn_network: The network the attractors were searched in
        fbm_attractors: The result of :func:`search_for_attractors`
        index: Which attractor (0-based) to draw
    """
    return draw_dynamic_for_one_matrix(fbn_network, attractor_to_matrix(fbm_attractors, index), seed=seed)


def plot_attractor_heatmap(fbm_attractors: Dict[str, Any], index: int = 0, ax=None, figsize=(8, 4)):
    """Draw one attractor cycle as a genes x steps on/off heatmap. Returns the matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    matrix = attractor_to_matrix(fbm_attractors, index)
    genes, cycle_length = list(matrix.index), matrix.shape[1]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(matrix.to_numpy(), cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_xticks(range(cycle_length))
    ax.set_xticklabels([str(i) for i in range(cycle_length)])
    for row in range(len(genes)):
        for col in range(cycle_length):
            value = int(matrix.iloc[row, col])
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value else "black", fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(cycle_length + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(genes) + 1)], minor=True)
    ax.grid(which="minor", color="lightgrey", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Step in cycle")
    ax.set_title(f"Attractor {index} (length {cycle_length})")
    return ax


_FORWARD_TYPES = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
_BACKWARD_TYPES = {1: 0, 2: 1}
_PLOT_TYPE = re.compile(r"^(forward|backward)_(\d)([ab])$")


def plot_network(
    fbn_network: Dict[str, Any],
    target_genes: Optional[List[str]] = None,
    type: str = "static",
    expand_level: int = 2,
    output_network: bool = False,
    timeseries_matrix: Optional[pd.DataFrame] = None,
    start_time_point: int = 1,
    end_time_point: int = 1,
    target_time_point: int = 1,
    seed: Optional[int] = None,
):
    """
    Plot a Fundamental Boolean Network in one call (R's ``plotNetwork``).

    ``type`` selects what is drawn:

    * ``"static"``, ``"staticSlice"``, ``"dynamic"`` - the (optionally gene filtered) network, the rules active at
      ``target_time_point``, or the rules active between ``start_time_point`` and ``end_time_point``.
    * ``"forward_<n><m>"`` - downstream targets of ``target_genes``. ``n`` selects the regulation: 1 = inhibitory
      rules acting on inhibited targets, 2 = inhibitory on activated, 3 = activating on inhibited, 4 = activating
      on activated. ``m`` is ``a`` (deeper levels keep the same regulation) or ``b`` (deeper levels use all rules).
    * ``"backward_<n><m>"`` - upstream regulators of ``target_genes``; ``n`` is 1 (inhibitors) or 2 (activators).

    Returns:
        The :class:`FBNGraph`; with ``output_network=True`` the tuple ``(network, graph)`` where ``network`` is the
        filtered Fundamental Boolean Network that was drawn.
    """
    _check_network(fbn_network)
    target_genes = list(target_genes or [])

    match = _PLOT_TYPE.match(type)
    if type in ("static", "staticSlice", "dynamic"):
        network = fbn_network
        if target_genes:
            network = filter_network_connections_by_genes(fbn_network, genelist=target_genes, exclusive=False, expand=False)
        graph = fbn_network_graph(
            network, type=type, timeseries_matrix=timeseries_matrix,
            from_time_point=start_time_point, to_time_point=target_time_point if type == "staticSlice" else end_time_point, seed=seed,
        )
    elif match and match.group(1) == "forward" and int(match.group(2)) in _FORWARD_TYPES:
        regulation, target = _FORWARD_TYPES[int(match.group(2))]
        network = find_forward_related_network_by_genes(
            fbn_network, target_gene_list=target_genes, regulation_type=regulation, target_type=target,
            max_deep=expand_level, next_level_mix_type=match.group(3) == "b")
        graph = fbn_network_graph(network)
    elif match and match.group(1) == "backward" and int(match.group(2)) in _BACKWARD_TYPES:
        network = find_backward_related_network_by_genes(
            fbn_network, target_gene_list=target_genes, regulation_type=_BACKWARD_TYPES[int(match.group(2))],
            max_deep=expand_level, next_level_mix_type=match.group(3) == "b")
        graph = fbn_network_graph(network)
    else:
        raise ValueError(
            f"Unknown type '{type}'. Use static, staticSlice, dynamic, forward_1a..forward_4b or backward_1a..backward_2b")

    return (network, graph) if output_network else graph
