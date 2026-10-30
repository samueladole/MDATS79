"""
RAGScope — Latency Chart Components
======================================
The pipeline flow Sankey: one query's stage-by-stage latency, drawn on the
Live Monitor and Query Explorer pages.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Fixed per-stage colours for the pipeline flow Sankey — a stage is always
# this same colour regardless of retrieval strategy, so e.g. "LLM Generation"
# reads the same whether the query used dense or hybrid retrieval.
#
# Colours are grouped by pipeline *phase* rather than assigned per-stage, so
# retrieval and generation are visually unmistakable at a glance: every
# retrieval sub-stage (embedding, search, fusion) is a shade of blue, LLM
# generation is amber, RAGAS evaluation is red — three clearly different
# hues, not a per-stage rainbow. Retrieval's own shades (light → dark) still
# let its individual sub-stages be told apart without breaking the grouping.
_NEUTRAL = "rgba(127,127,127,0.55)"
_RETRIEVAL_EMBED = "#7fb3e8"   # lightest blue — query embedding
_RETRIEVAL_SEARCH = "#2a78d6"  # mid blue — vector / dense search
_RETRIEVAL_SPARSE = "#1f5fae"  # darker blue — BM25 sparse search
_RETRIEVAL_FUSION = "#123a70"  # darkest blue — RRF fusion
_GENERATION = "#eda100"        # amber — clearly distinct from the blue family
_EVALUATION = "#e34948"        # red/coral

STAGE_COLORS = {
    "Query": _NEUTRAL,
    "Embed Query": _RETRIEVAL_EMBED,
    "Vector Search": _RETRIEVAL_SEARCH,
    "Dense Search": _RETRIEVAL_SEARCH,
    "BM25 Search": _RETRIEVAL_SPARSE,
    "RRF Fusion": _RETRIEVAL_FUSION,
    "LLM Generation": _GENERATION,
    "RAGAS Evaluation": _EVALUATION,
    "Response": _NEUTRAL,
}

# Which phase each stage belongs to — drives the small phase caption on each
# Sankey node and the legend below the chart. Query/Response sit outside any
# phase (they're the pipeline's start/overhead-catch-all, not work RAGScope
# attributes to retrieval, generation, or evaluation).
STAGE_PHASE = {
    "Embed Query": "RETRIEVAL",
    "Vector Search": "RETRIEVAL",
    "Dense Search": "RETRIEVAL",
    "BM25 Search": "RETRIEVAL",
    "RRF Fusion": "RETRIEVAL",
    "Entity Seeding": "RETRIEVAL",
    "Graph Traversal": "RETRIEVAL",
    "Chunk Scoring": "RETRIEVAL",
    "LLM Generation": "GENERATION",
    "RAGAS Evaluation": "EVALUATION",
}


def _translucent(color: str, alpha: float = 0.35) -> str:
    """Return ``color`` (hex or rgba(...)) at a reduced alpha, for Sankey links."""
    if color.startswith("rgba"):
        r, g, b = (p.strip() for p in color[color.index("(") + 1 : color.index(")")].split(",")[:3])
        return f"rgba({r},{g},{b},{alpha})"
    hex_color = color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _is_dark_theme() -> bool:
    """Best-effort detection of the viewer's active Streamlit theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def pipeline_flow_sankey(
    retrieval_telemetry: dict,
    generation_ms: float,
    evaluation_ms: float,
    e2e_ms: float,
    ran_evaluation: bool,
) -> None:
    """
    Render a Sankey diagram of one query's full pipeline execution.

    Every instrumented stage — query embedding, retrieval (dense, or
    dense+BM25+RRF fusion for hybrid), LLM generation, and RAGAS evaluation —
    appears as a node; each link's width is that stage's actual wall-clock
    duration for *this* query, not an aggregate. Stages run sequentially in
    the current implementation (matching how they're actually instrumented
    in ``pipeline/retrieval/*.py``), so this is a straight chain rather than
    parallel branches, even for hybrid retrieval where BM25 search doesn't
    itself depend on the query embedding.

    Parameters
    ----------
    retrieval_telemetry : The ``retrieval_telemetry`` dict from a ``RAGResult``
                          (or an equivalent dict reconstructed from a stored
                          telemetry record).
    generation_ms       : LLM generation wall-clock time.
    evaluation_ms       : RAGAS evaluation wall-clock time (0 if skipped).
    e2e_ms              : Total end-to-end wall-clock time for the query.
    ran_evaluation      : Whether RAGAS evaluation actually ran — when False,
                          the Evaluation node is omitted entirely rather than
                          shown as a (misleading) near-zero-width stage.
    """
    strategy = retrieval_telemetry.get("strategy", "dense")

    stages: list[tuple[str, float]] = [
        ("Query", 0.0),
        ("Embed Query", retrieval_telemetry.get("embed_query_ms", 0.0)),
    ]
    if strategy == "hybrid":
        stages += [
            ("Dense Search", retrieval_telemetry.get("dense_search_ms", 0.0)),
            ("BM25 Search", retrieval_telemetry.get("bm25_search_ms", 0.0)),
            ("RRF Fusion", retrieval_telemetry.get("rrf_fusion_ms", 0.0)),
        ]
    elif strategy == "graph":
        stages += [
            ("Dense Search", retrieval_telemetry.get("dense_search_ms", 0.0)),
            ("Entity Seeding", retrieval_telemetry.get("graph_seed_ms", 0.0)),
            ("Graph Traversal", retrieval_telemetry.get("graph_traverse_ms", 0.0)),
            ("Chunk Scoring", retrieval_telemetry.get("graph_score_ms", 0.0)),
            ("RRF Fusion", retrieval_telemetry.get("rrf_fusion_ms", 0.0)),
        ]
    else:
        stages.append(("Vector Search", retrieval_telemetry.get("vector_search_ms", 0.0)))

    stages.append(("LLM Generation", generation_ms))
    if ran_evaluation:
        stages.append(("RAGAS Evaluation", evaluation_ms))

    # The final link absorbs whatever's left of e2e_ms after the named stages
    # (token counting, telemetry write, python glue) — kept visible rather
    # than silently dropped, so the diagram's total always equals e2e_ms.
    accounted = sum(v for _, v in stages)
    overhead = max(e2e_ms - accounted, 0.0)
    stages.append(("Response", overhead))

    total = e2e_ms if e2e_ms > 0 else max(accounted, 1.0)
    dark = _is_dark_theme()
    # Plotly's Sankey renders to a self-contained SVG canvas — it does not
    # inherit the page's CSS text colour, so with a transparent chart
    # background the default label colour goes illegible against a dark
    # Streamlit theme. Pin it explicitly based on the detected theme instead.
    label_color = "#f5f5f2" if dark else "#1a1a1a"
    phase_label_color = "rgba(245,245,242,0.65)" if dark else "rgba(26,26,26,0.6)"
    node_line_color = "rgba(255,255,255,0.30)" if dark else "rgba(0,0,0,0.20)"

    # Each node's label carries its phase (RETRIEVAL / GENERATION / EVALUATION)
    # as a small caption line above the stage name — on top of the phase
    # colour-coding below, so which phase a stage belongs to is legible even
    # without colour (e.g. on hover, or for colour-blind readers).
    labels = []
    for name, value in stages:
        phase = STAGE_PHASE.get(name)
        phase_line = (
            f"<span style='font-size:9px;letter-spacing:0.05em;color:{phase_label_color}'>{phase}</span><br>"
            if phase
            else ""
        )
        labels.append(f"{phase_line}<b>{name}</b><br>{value:,.0f} ms · {value / total * 100:.1f}%")
    node_colors = [STAGE_COLORS.get(name, "#95a5a6") for name, _ in stages]

    n = len(stages)
    sources = list(range(n - 1))
    targets = list(range(1, n))
    # Link i (stage[i] -> stage[i+1]) is coloured and sized by the stage it
    # leads into, so a link's width directly reads as "how long that stage took".
    values = [max(stages[i + 1][1], 0.05) for i in range(n - 1)]
    link_colors = [_translucent(node_colors[i + 1], alpha=0.45) for i in range(n - 1)]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=node_colors,
                pad=30,
                thickness=24,
                line=dict(color=node_line_color, width=1),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="<b>%{target.label}</b><extra></extra>",
            ),
            textfont=dict(size=13, color=label_color, family="system-ui, -apple-system, sans-serif"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=label_color),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # A colour-swatch legend under the chart makes the retrieval/generation/
    # evaluation grouping explicit even before a viewer reads any node label.
    def _swatch(color: str, text: str) -> str:
        return (
            f'<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:1.1rem;">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:0.8rem;opacity:0.8;">{text}</span></span>'
        )

    st.markdown(
        '<div style="margin-top:-0.5rem;">'
        + _swatch(_RETRIEVAL_SEARCH, "Retrieval (embed → search → fusion)")
        + _swatch(_GENERATION, "Generation (LLM)")
        + _swatch(_EVALUATION, "Evaluation (RAGAS)")
        + "</div>",
        unsafe_allow_html=True,
    )
