"""
Interactive retrieval-path graph for the graph retrieval strategy.

Renders the ``graph_path`` produced by ``GraphRetriever`` as a layered,
left-to-right directed acyclic graph:

    Query  ->  seed entities  ->  hop 1  ->  hop 2  ->  retrieved chunks

Columns are hop distance from the seed entities, and every arrow points one
column to the right, so the retrieval paths form a DAG by construction (hop
distance strictly increases along a shortest path). Relations that run between
two nodes in the same column are context, not part of any path, and are drawn
as plain dotted lines without arrowheads.

Shared by Live Monitor (the query just run) and Query Explorer (a persisted
record), so both pages draw exactly the same picture of the same data.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Entity colours follow the dashboard's existing palette; the retrieved chunk is
# green so a reader's eye goes to where every path ends.
TYPE_COLORS = {
    "ChemicalOrDrug": "#eda100",  # amber
    "GeneOrProtein": "#2a78d6",   # mid blue
    "Disease": "#e34948",         # coral
}
TYPE_NAMES = {
    "ChemicalOrDrug": "Chemical / drug",
    "GeneOrProtein": "Gene / protein",
    "Disease": "Disease",
}
FALLBACK_ENTITY = "#8e6fd8"
QUERY_COLOR = "rgba(127,127,127,0.95)"
DOC_COLOR = "rgba(127,127,127,0.75)"
CHUNK_COLOR = "#24a35a"

PATH_EDGE = "rgba(42,120,214,0.85)"
PATH_EDGE_FADED = "rgba(42,120,214,0.18)"
HIGHLIGHT_EDGE = CHUNK_COLOR
CONTEXT_EDGE = "rgba(127,127,127,0.40)"

# Short labels written on path edges, read in the direction of the arrow; the
# full relation in its stored direction is in the hover. When the walk ran
# against a stored relation the reversed phrasing is used, so the arrow and
# its label never contradict each other ("protein -> chunk: mentioned in",
# not "mentions").
EDGE_SHORT = {
    "SEED": "match",
    "MENTIONS_COMPOUND": "mentions",
    "MENTIONS_DISEASE": "mentions",
    "MENTIONS_GENE": "mentions",
    "INHIBITS": "inhibits",
    "TREATS": "treats",
    "EXTRACTED_FROM": "source",
}
EDGE_SHORT_REVERSED = {
    "MENTIONS_COMPOUND": "mentioned in",
    "MENTIONS_DISEASE": "mentioned in",
    "MENTIONS_GENE": "mentioned in",
    "INHIBITS": "inhibited by",
    "TREATS": "treated by",
}


def _short_label(edge: dict) -> str:
    label = edge["label"]
    if edge.get("reversed"):
        return EDGE_SHORT_REVERSED.get(label, f"{label.lower().replace('_', ' ')} (reversed)")
    return EDGE_SHORT.get(label, label.lower().replace("_", " "))


# ── Layout ────────────────────────────────────────────────────────────────────


def _layout(nodes: list[dict], edges: list[dict]) -> dict[str, tuple[float, float]]:
    """Place nodes in columns by layer, ordered to reduce edge crossings.

    A single barycentric sweep (the ordering step of Sugiyama layout): each
    node is placed at the mean height of its neighbours in the previous
    column, with PageRank probability breaking ties. Cheap, deterministic, and
    good enough for graphs of a few dozen nodes.
    """
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        by_layer[n["layer"]].append(n)

    neighbours: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        neighbours[e["source"]].add(e["target"])
        neighbours[e["target"]].add(e["source"])

    pos: dict[str, tuple[float, float]] = {}
    for layer in sorted(by_layer):
        col = by_layer[layer]

        def barycentre(n: dict) -> float:
            ys = [pos[m][1] for m in neighbours[n["id"]] if m in pos]
            return sum(ys) / len(ys) if ys else 0.0

        col.sort(key=lambda n: (barycentre(n), -n.get("ppr", 0.0), n["label"]))
        span = len(col) - 1
        for i, n in enumerate(col):
            pos[n["id"]] = (float(layer), (i - span / 2) * 1.0)
    return pos


def _node_size(n: dict, max_ppr: float) -> float:
    if n["kind"] == "query":
        return 22
    if n["kind"] == "chunk":
        return 24
    if n["kind"] == "document":
        return 12
    share = (n.get("ppr", 0.0) / max_ppr) if max_ppr > 0 else 0.0
    return 13 + 25 * math.sqrt(share)


def _truncate(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _hover(n: dict, chunk_texts: dict[str, str]) -> str:
    kind = n["kind"]
    prob = f"{100 * n.get('ppr', 0.0):.2f}%"
    if kind == "query":
        return f"<b>Query</b><br>{_truncate(n['label'], 160)}"
    if kind == "chunk":
        snippet = chunk_texts.get(n.get("chunk_id", ""), "")
        origin = ("found only by the graph" if n.get("from_graph_leg")
                  else "also found by dense retrieval")
        lines = [
            f"<b>Retrieved chunk #{n.get('result_rank', '?')}</b>",
            n.get("chunk_id", ""),
            f"Graph score: {n.get('graph_score', 0.0):.5f}  (graph rank {n.get('graph_rank', '?')})",
            f"Fused RRF score: {n.get('rrf_score', 0.0):.5f}",
            f"PageRank probability: {prob}",
            f"Origin: {origin}",
        ]
        if snippet:
            lines.append("<i>" + _truncate(snippet, 160).replace("\n", " ") + "</i>")
        return "<br>".join(lines)
    if kind == "document":
        return (f"<b>Document {n['label']}</b><br>"
                "Co-occurrence waypoint — the path passes through this chunk's "
                "entities, but the chunk itself is not in the result set"
                f"<br>PageRank probability: {prob}")
    lines = [
        f"<b>{n['label']}</b>",
        TYPE_NAMES.get(n.get("type", ""), n.get("type", "Entity")),
        f"PageRank probability: {prob}",
    ]
    if kind == "seed":
        lines.append(
            f"Seed entity — matched by {n.get('matcher', '?')} "
            f"({n.get('match_score', 0.0):.3f})"
        )
    if "n_chunks" in n:
        lines.append(f"Appears in {n['n_chunks']} chunk(s) in the graph")
    if not n.get("on_path"):
        lines.append("<i>Context only — not on any retrieval path</i>")
    return "<br>".join(lines)


# ── Figure ────────────────────────────────────────────────────────────────────


def retrieval_graph_figure(
    graph_path: dict,
    *,
    show_context: bool = True,
    highlight_chunk: str | None = None,
    chunk_texts: dict[str, str] | None = None,
) -> go.Figure:
    """Build the interactive retrieval-path figure.

    Parameters
    ----------
    graph_path      : The ``graph_path`` telemetry produced by GraphRetriever.
    show_context    : Include off-path context entities and their links.
    highlight_chunk : chunk_id whose path is emphasised; others are faded.
    chunk_texts     : {chunk_id: text}, shown in the hover of chunk nodes.
    """
    chunk_texts = chunk_texts or {}
    nodes = [n for n in graph_path.get("nodes", [])
             if show_context or n.get("on_path") or n["kind"] == "query"]
    keep = {n["id"] for n in nodes}
    edges = [e for e in graph_path.get("edges", [])
             if e["source"] in keep and e["target"] in keep
             and (show_context or e.get("on_path"))]
    by_id = {n["id"]: n for n in nodes}
    pos = _layout(nodes, edges)
    max_ppr = max((n.get("ppr", 0.0) for n in nodes), default=0.0)

    highlight_nodes: set[str] = set()
    highlight_edges: set[tuple[str, str]] = set()
    if highlight_chunk:
        for p in graph_path.get("paths", []):
            if p["chunk_id"] == highlight_chunk:
                seq = ["__query__", *p["nodes"]]
                highlight_nodes = set(seq)
                highlight_edges = set(zip(seq, seq[1:], strict=False))

    fig = go.Figure()

    # Context links: one trace, dotted, no arrowheads.
    cx, cy = [], []
    for e in edges:
        if e.get("on_path"):
            continue
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        cx += [x0, x1, None]
        cy += [y0, y1, None]
    if cx:
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines", name="Context links",
            line=dict(color=CONTEXT_EDGE, width=1, dash="dot"),
            hoverinfo="skip", legendgroup="context",
        ))

    # Path links: drawn as arrow annotations so direction is explicit.
    annotations = []
    mid_x, mid_y, mid_hover, mid_text, mid_ids = [], [], [], [], []
    for i, e in enumerate(edges):
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        key = (e["source"], e["target"])
        if e.get("on_path"):
            hi = key in highlight_edges
            faded = bool(highlight_chunk) and not hi
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.1,
                arrowwidth=3.5 if hi else 2,
                arrowcolor=HIGHLIGHT_EDGE if hi else (PATH_EDGE_FADED if faded else PATH_EDGE),
                standoff=_node_size(by_id[e["target"]], max_ppr) / 2 + 2,
                startstandoff=_node_size(by_id[e["source"]], max_ppr) / 2 + 2,
                text="",
            ))
        # Labels sit 40% along the edge rather than at its midpoint, keeping
        # them clear of the node labels in the destination column.
        mid_x.append(x0 + 0.4 * (x1 - x0))
        mid_y.append(y0 + 0.4 * (y1 - y0))
        mid_hover.append(
            f"<b>{e['label']}</b><br>{e.get('relation', '')}"
            + ("" if e.get("on_path") else "<br><i>Context — not on a retrieval path</i>")
            + ("<br><i>Same hop distance — no traversal direction</i>"
               if e.get("same_layer") else "")
        )
        mid_text.append(_short_label(e) if e.get("on_path") else "")
        mid_ids.append(f"edge::{i}")

    # Edge midpoints: carry the relation hover and the short on-edge label.
    if mid_x:
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y, mode="markers+text", name="Relations",
            text=mid_text, textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=9, color="rgba(127,127,127,0.05)"),
            hovertext=mid_hover, hoverinfo="text",
            customdata=mid_ids, showlegend=False,
        ))

    # Nodes: one trace per COLOUR, so the legend is a true colour key and each
    # entry toggles exactly the nodes drawn in that colour. Role is carried
    # per point instead — seeds by a heavy dark outline, off-path context by
    # reduced opacity — because a legend swatch can show only one colour, and
    # grouping seeds into their own trace made it show the wrong one.
    groups: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        if n["kind"] == "query":
            groups["Query"].append(n)
        elif n["kind"] == "chunk":
            groups["Retrieved chunk"].append(n)
        elif n["kind"] == "document":
            groups["Document (waypoint)"].append(n)
        else:
            groups[TYPE_NAMES.get(n.get("type", ""), "Entity")].append(n)

    # Seed halo, drawn beneath the nodes. A mid-grey ring stays visible on both
    # light and dark themes, where any single dark outline disappears against a
    # dark background and a light one against a light background.
    seeds = [n for n in nodes if n["kind"] == "seed"]
    if seeds:
        fig.add_trace(go.Scatter(
            x=[pos[n["id"]][0] for n in seeds],
            y=[pos[n["id"]][1] for n in seeds],
            mode="markers", name="Seed entity", showlegend=True,
            marker=dict(
                size=[_node_size(n, max_ppr) + 12 for n in seeds],
                # For open symbols Plotly strokes the ring in marker.color;
                # marker.line only sets its width. A transparent marker.color
                # draws nothing at all.
                color="rgba(150,150,150,0.95)", symbol="circle-open",
                line=dict(width=3),
                opacity=[0.3 if (highlight_chunk and n["id"] not in highlight_nodes) else 1.0
                         for n in seeds],
            ),
            hoverinfo="skip",
        ))

    order = ["Query", "Chemical / drug", "Gene / protein", "Disease", "Entity",
             "Document (waypoint)", "Retrieved chunk"]
    for name in order:
        members = groups.get(name)
        if not members:
            continue
        kind = members[0]["kind"]
        colours = [
            QUERY_COLOR if m["kind"] == "query"
            else CHUNK_COLOR if m["kind"] == "chunk"
            else DOC_COLOR if m["kind"] == "document"
            else TYPE_COLORS.get(m.get("type", ""), FALLBACK_ENTITY)
            for m in members
        ]
        opacity = [
            0.3 if (highlight_chunk and m["id"] not in highlight_nodes)
            else (0.5 if not m.get("on_path") else 1.0)
            for m in members
        ]
        symbol = {"query": "diamond", "chunk": "square",
                  "document": "square-open"}.get(kind, "circle")
        # Chunk labels sit beneath the square rather than beside it, so a long
        # chunk id in the right-most column cannot run off the plot edge.
        textpos = {"query": "bottom center", "chunk": "bottom center"}.get(kind, "top center")

        def node_text(m: dict) -> str:
            if m["kind"] == "query":
                return "Query"
            if m["kind"] == "chunk":
                return f"#{m.get('result_rank', '?')} · {m.get('chunk_id', '')}"
            if m["kind"] == "document":
                return _truncate(m["label"], 24)
            return _truncate(m["label"], 24)

        fig.add_trace(go.Scatter(
            x=[pos[m["id"]][0] for m in members],
            y=[pos[m["id"]][1] for m in members],
            mode="markers+text",
            name=name,
            text=[node_text(m) for m in members],
            textposition=textpos,
            textfont=dict(size=11),
            marker=dict(
                size=[_node_size(m, max_ppr) for m in members],
                color=colours,
                opacity=opacity,
                symbol=symbol,
                line=dict(
                    # Seeds get a heavy dark outline so the walk's starting
                    # points stand out whatever their entity type.
                    width=[3.5 if m["kind"] == "seed" else 1 for m in members],
                    color=["rgba(20,20,20,0.9)" if m["kind"] == "seed"
                           else "rgba(255,255,255,0.6)" for m in members],
                ),
            ),
            hovertext=[_hover(m, chunk_texts) for m in members],
            hoverinfo="text",
            customdata=[m["id"] for m in members],
            legendgroup=name,
        ))

    # Column headers make the hop structure legible at a glance.
    layers = sorted({n["layer"] for n in nodes})
    ys = [p[1] for p in pos.values()] or [0.0]
    top = max(ys) + 0.75
    for layer in layers:
        header = {0: "Query", 1: "Seed entities"}.get(layer, f"Hop {layer - 1}")
        annotations.append(dict(
            x=layer, y=top, xref="x", yref="y", text=f"<b>{header}</b>",
            showarrow=False, font=dict(size=11, color="rgba(127,127,127,0.95)"),
        ))

    tallest = max((sum(1 for n in nodes if n["layer"] == L) for L in layers), default=1)
    fig.update_layout(
        # The full query is the title, rather than a node label, because a
        # long query cannot fit beside a node without being clipped.
        title=dict(
            text=f"<b>Query:</b> {_truncate(graph_path.get('query', ''), 120)}",
            font=dict(size=13), x=0, xanchor="left", y=0.99, yanchor="top",
        ),
        annotations=annotations,
        # Generous vertical spacing: the dashboard's centred layout gives the
        # figure only ~700px of width, so labels need room to avoid colliding.
        height=max(380, 80 * tallest + 170),
        margin=dict(l=10, r=10, t=86, b=10),
        hovermode="closest",
        dragmode="pan",
        clickmode="event+select",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis=dict(visible=False, range=[min(layers) - 0.45, max(layers) + 0.95]),
        # An explicit range removes the dead space autorange left below the graph.
        yaxis=dict(visible=False, range=[min(ys) - 0.6, top + 0.35]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Tabular companion ─────────────────────────────────────────────────────────


def path_table_rows(graph_path: dict) -> list[dict]:
    """One row per graph-reached chunk: the path the walk took to reach it."""
    rows = []
    for p in sorted(graph_path.get("paths", []), key=lambda p: p["result_rank"]):
        rows.append({
            "Rank": p["result_rank"],
            "Chunk": p["chunk_id"],
            "Retrieval path": "Query → " + " → ".join(p["labels"]),
            "Hops": p["hops"],
            "Graph score": round(p["graph_score"], 5),
            "Final-hop probability": f"{100 * p['contributor_ppr']:.2f}%",
            "Origin": "graph only" if p["from_graph_leg"] else "graph + dense",
        })
    return rows


# ── Streamlit renderer ────────────────────────────────────────────────────────


def _selected_ids(event) -> list[str]:
    """Node/edge ids from a plotly selection event, tolerant of payload shape."""
    try:
        points = event.selection.points  # type: ignore[union-attr]
    except AttributeError:
        try:
            points = event["selection"]["points"]  # type: ignore[index]
        except (KeyError, TypeError):
            return []
    out = []
    for p in points or []:
        cd = p.get("customdata") if isinstance(p, dict) else None
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        if cd:
            out.append(str(cd))
    return out


def render_retrieval_graph(
    graph_path: dict | None,
    *,
    chunk_texts: dict[str, str] | None = None,
    key: str,
) -> None:
    """Draw the interactive retrieval graph plus its path table.

    ``key`` must be unique per page and per record, since Streamlit keys the
    figure's selection state on it.
    """
    if not graph_path:
        st.info(
            "No retrieval graph for this query: no entity in it matched the "
            "knowledge graph, so there was no walk to draw."
        )
        return

    chunk_texts = chunk_texts or {}
    paths = graph_path.get("paths", [])

    c1, c2 = st.columns([1, 2])
    show_context = c1.toggle(
        "Show context entities", value=True, key=f"{key}_ctx",
        help="High-probability entities next to the paths but not on them.",
    )
    options = ["All paths"] + [f"#{p['result_rank']} · {p['chunk_id']}" for p in paths]
    choice = c2.selectbox(
        "Highlight the path to", options, key=f"{key}_hl",
        help="Emphasise the walk that reached one chunk and fade the rest.",
    )
    highlight = None if choice == "All paths" else paths[options.index(choice) - 1]["chunk_id"]

    fig = retrieval_graph_figure(
        graph_path, show_context=show_context,
        highlight_chunk=highlight, chunk_texts=chunk_texts,
    )
    event = st.plotly_chart(
        fig, width="stretch", key=f"{key}_fig",
        on_select="rerun", selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "Read left to right: the query matches **seed entities** (heavy dark "
        "outline), the walk expands one hop per column, and every arrow ends one "
        "column further right, so the paths form a directed acyclic graph. Colour "
        "is entity type; faded nodes are context, near the paths but not on them. "
        "Node size is "
        "**PageRank probability** — "
        + graph_path.get("probability_note", "").replace("Node probability is ", "").rstrip(".")
        + ". Arrows show the order the walk took; hover an edge for the relation "
        "in its true direction. Click a node for details; drag to pan, scroll to zoom, "
        "and click legend entries to hide categories."
    )

    # Click-through detail for the selected node.
    by_id = {n["id"]: n for n in graph_path.get("nodes", [])}
    for sel in _selected_ids(event):
        node = by_id.get(sel)
        if not node:
            continue
        with st.container(border=True):
            if node["kind"] == "chunk":
                st.markdown(f"**Retrieved chunk #{node.get('result_rank')}** · `{node.get('chunk_id')}`")
                m1, m2, m3 = st.columns(3)
                m1.metric("Graph score", f"{node.get('graph_score', 0.0):.5f}")
                m2.metric("Fused RRF score", f"{node.get('rrf_score', 0.0):.5f}")
                m3.metric("PageRank probability", f"{100 * node.get('ppr', 0.0):.2f}%")
                path = next((p for p in paths if p["chunk_id"] == node.get("chunk_id")), None)
                if path:
                    st.markdown("Reached by: " + " → ".join(["Query", *path["labels"]]))
                text = chunk_texts.get(node.get("chunk_id", ""), "")
                if text:
                    st.text(text[:1200] + ("…" if len(text) > 1200 else ""))
            elif node["kind"] == "query":
                st.markdown(f"**Query** · {node['label']}")
            else:
                st.markdown(
                    f"**{node['label']}** · "
                    f"{TYPE_NAMES.get(node.get('type', ''), node.get('type', ''))}"
                )
                m1, m2 = st.columns(2)
                m1.metric("PageRank probability", f"{100 * node.get('ppr', 0.0):.2f}%")
                if "n_chunks" in node:
                    m2.metric("Chunks containing it", node["n_chunks"])
                via = [p for p in paths if sel in p["nodes"]]
                if via:
                    st.markdown(
                        "On the path to: "
                        + ", ".join(f"#{p['result_rank']} `{p['chunk_id']}`" for p in via)
                    )
                elif node["kind"] != "query":
                    st.caption("Context only — the walk passed nearby but no "
                               "retrieved chunk was reached through this node.")

    st.markdown("**How the graph reached each chunk**")
    rows = path_table_rows(graph_path)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption(
            "The walk found seed entities but none of the returned chunks were "
            "reached through the graph — in fused mode, the dense leg supplied them all."
        )


# ── Entity explorer ───────────────────────────────────────────────────────────
#
# The Knowledge Base page's view of the graph itself, as opposed to one query's
# walk through it. Same visual language as the retrieval graph above: columns
# read left to right, colour is entity type, green squares are chunks, and
# every arrow points one or more columns to the right, so the drawing is a DAG.
#
#     focus entity  ->  related entities  ->  source chunks

DOC_PREFIX = "DOC::"
SOURCE_SHORT = "in"  # entity -> chunk it was extracted from


def _entity_name(digraph, nid: str) -> str:
    data = digraph.nodes.get(nid, {}) if nid in digraph else {}
    return str(data.get("name") or data.get("label") or nid)


def entity_neighbourhood(
    digraph,
    node_chunks: dict[str, list[str]],
    focus: str,
    *,
    max_neighbours: int = 12,
    max_chunks: int = 14,
    chunks_per_neighbour: int = 2,
) -> dict:
    """One entity's neighbourhood as a three-column DAG.

    Column 0 is ``focus``; column 1 the entities it shares a relation with,
    most-connected first; column 2 the chunks those entities were extracted
    from, the focus entity's own chunks first. Every edge points rightwards
    and carries the relation in reading direction, with ``reversed`` set when
    the stored relation runs the other way (so "aspirin -> COX-1: inhibits"
    but "COX-1 -> aspirin: inhibited by").
    """
    if focus not in digraph:
        return {"focus": focus, "nodes": [], "edges": []}

    def entity_node(nid: str, layer: int, kind: str) -> dict:
        data = digraph.nodes[nid]
        return {"id": nid, "label": _entity_name(digraph, nid), "kind": kind,
                "type": str(data.get("__class__", "Entity")), "layer": layer,
                "n_chunks": len(node_chunks.get(nid, []))}

    neighbours = {
        n for n in set(digraph.successors(focus)) | set(digraph.predecessors(focus))
        if not str(n).startswith(DOC_PREFIX) and n != focus
    }
    ranked = sorted(neighbours, key=lambda n: (-len(node_chunks.get(n, [])),
                                               _entity_name(digraph, n).lower()))
    shown = ranked[:max_neighbours]

    nodes = [entity_node(focus, 0, "focus")] + [entity_node(n, 1, "entity") for n in shown]
    edges = []
    for n in shown:
        if digraph.has_edge(focus, n):
            label, rev = digraph.edges[focus, n].get("label", "RELATED_TO"), False
        else:
            label, rev = digraph.edges[n, focus].get("label", "RELATED_TO"), True
        edges.append({"source": focus, "target": n, "label": label, "reversed": rev,
                      "relation": (f"{_entity_name(digraph, n)} —{label}→ {_entity_name(digraph, focus)}"
                                   if rev else
                                   f"{_entity_name(digraph, focus)} —{label}→ {_entity_name(digraph, n)}")})

    # Chunks: the focus entity's own first, then a few per related entity, so
    # shared chunks (co-occurrence) show up as squares with several arrows in.
    chunks: list[str] = []
    for cid in node_chunks.get(focus, []):
        if cid not in chunks:
            chunks.append(cid)
    for n in shown:
        for cid in node_chunks.get(n, [])[:chunks_per_neighbour]:
            if cid not in chunks:
                chunks.append(cid)
    chunks = chunks[:max_chunks]
    hidden_chunks = len(set(node_chunks.get(focus, []))) - sum(
        1 for c in chunks if c in set(node_chunks.get(focus, [])))

    for cid in chunks:
        nodes.append({"id": f"{DOC_PREFIX}{cid}", "label": cid, "kind": "source",
                      "type": "Source chunk", "layer": 2, "chunk_id": cid})
    for ent in [focus, *shown]:
        for cid in node_chunks.get(ent, []):
            if cid in chunks:
                edges.append({"source": ent, "target": f"{DOC_PREFIX}{cid}",
                              "label": "EXTRACTED_FROM", "reversed": False,
                              "relation": f"{_entity_name(digraph, ent)} was extracted from {cid}"})

    return {
        "focus": focus,
        "nodes": nodes,
        "edges": edges,
        "hidden_neighbours": max(0, len(ranked) - len(shown)),
        "hidden_chunks": max(0, hidden_chunks),
    }


def entity_graph_figure(view: dict) -> go.Figure:
    """Draw ``entity_neighbourhood`` output."""
    nodes, edges = view["nodes"], view["edges"]
    pos = _layout(nodes, edges)
    by_id = {n["id"]: n for n in nodes}
    max_chunks = max((n.get("n_chunks", 0) for n in nodes), default=0)

    def size(n: dict) -> float:
        if n["kind"] == "source":
            return 18
        share = (n.get("n_chunks", 0) / max_chunks) if max_chunks else 0.0
        return (30 if n["kind"] == "focus" else 13) + 16 * math.sqrt(share)

    fig = go.Figure()
    annotations = []
    mid_x, mid_y, mid_text, mid_hover = [], [], [], []
    for e in edges:
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        is_source = e["label"] == "EXTRACTED_FROM"
        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.0,
            arrowwidth=1.4 if is_source else 2,
            arrowcolor=CONTEXT_EDGE if is_source else PATH_EDGE,
            standoff=size(by_id[e["target"]]) / 2 + 2,
            startstandoff=size(by_id[e["source"]]) / 2 + 2, text="",
        ))
        if not is_source:  # relation labels only; "extracted from" is implied by the column
            mid_x.append(x0 + 0.45 * (x1 - x0))
            mid_y.append(y0 + 0.45 * (y1 - y0))
            mid_text.append(_short_label(e))
            mid_hover.append(f"<b>{e['label']}</b><br>{e['relation']}")
    if mid_x:
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y, mode="markers+text", text=mid_text,
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=9, color="rgba(127,127,127,0.05)"),
            hovertext=mid_hover, hoverinfo="text", showlegend=False, name="Relations",
        ))

    groups: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        groups["Source chunk" if n["kind"] == "source"
               else TYPE_NAMES.get(n["type"], "Entity")].append(n)

    for name in ["Chemical / drug", "Gene / protein", "Disease", "Entity", "Source chunk"]:
        members = groups.get(name)
        if not members:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[m["id"]][0] for m in members],
            y=[pos[m["id"]][1] for m in members],
            mode="markers+text", name=name,
            text=[_truncate(m["label"], 26) for m in members],
            textposition=["bottom center" if m["kind"] == "source" else "top center"
                          for m in members],
            textfont=dict(size=11),
            marker=dict(
                size=[size(m) for m in members],
                color=[CHUNK_COLOR if m["kind"] == "source"
                       else TYPE_COLORS.get(m["type"], FALLBACK_ENTITY) for m in members],
                symbol=["square" if m["kind"] == "source" else "circle" for m in members],
                line=dict(
                    width=[3.5 if m["kind"] == "focus" else 1 for m in members],
                    color=["rgba(20,20,20,0.9)" if m["kind"] == "focus"
                           else "rgba(255,255,255,0.6)" for m in members],
                ),
            ),
            hovertext=[
                (f"<b>Source chunk</b><br>{m['chunk_id']}<br><i>Click to read it</i>")
                if m["kind"] == "source" else
                (f"<b>{m['label']}</b><br>{TYPE_NAMES.get(m['type'], m['type'])}"
                 f"<br>Extracted from {m.get('n_chunks', 0)} chunk(s)"
                 + ("<br><i>Focus entity</i>" if m["kind"] == "focus"
                    else "<br><i>Click to explore it</i>"))
                for m in members
            ],
            hoverinfo="text", customdata=[m["id"] for m in members], legendgroup=name,
        ))

    layers = sorted({n["layer"] for n in nodes}) or [0]
    ys = [p[1] for p in pos.values()] or [0.0]
    top = max(ys) + 0.8
    for layer, header in ((0, "Entity"), (1, "Related entities"), (2, "Source chunks")):
        if layer in layers:
            annotations.append(dict(
                x=layer, y=top, xref="x", yref="y", text=f"<b>{header}</b>",
                showarrow=False, font=dict(size=11, color="rgba(127,127,127,0.95)"),
            ))
    tallest = max((sum(1 for n in nodes if n["layer"] == L) for L in layers), default=1)
    fig.update_layout(
        annotations=annotations,
        height=max(360, 62 * tallest + 140),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest", dragmode="pan", clickmode="event+select",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis=dict(visible=False, range=[min(layers) - 0.45, max(layers) + 0.6]),
        yaxis=dict(visible=False, range=[min(ys) - 0.7, top + 0.35]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_entity_graph(view: dict, *, key: str) -> str | None:
    """Draw the entity explorer and return the id of a clicked node, if any."""
    if not view["nodes"]:
        st.info("Entity not found in the graph.")
        return None
    event = st.plotly_chart(
        entity_graph_figure(view), width="stretch", key=f"{key}_fig",
        on_select="rerun", selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True},
    )
    notes = []
    if view.get("hidden_neighbours"):
        notes.append(f"{view['hidden_neighbours']} more related entities not shown")
    if view.get("hidden_chunks"):
        notes.append(f"{view['hidden_chunks']} more source chunks not shown")
    st.caption(
        "Read left to right: the entity, the entities it has a relation with, and "
        "the chunks they were extracted from. A chunk with several arrows in is "
        "where entities co-occur. Click an entity to explore it, or a chunk to "
        "read it; drag to pan and scroll to zoom."
        + (f" ({'; '.join(notes)}.)" if notes else "")
    )
    ids = _selected_ids(event)
    return ids[0] if ids else None
