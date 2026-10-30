"""
RAGScope — Pipeline Flow Chart
================================
A Sankey diagram of where one query's time went, drawn on the Live Monitor
and Query Explorer pages.

It reads left to right as a whole splitting into its parts:

    End to end  ->  Retrieval / Generation / Evaluation / Other  ->  retrieval steps

so every band's width is a share of the query's total time, and the bands
leaving a node add up to that node. Node positions are fixed rather than left
to Plotly, so labels never collide however small a stage is.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.formatting import format_duration

# Colour by phase: retrieval in blues (light to dark through its steps),
# generation amber, evaluation coral. The same stage is always the same colour.
TOTAL_COLOR = "#6b7280"
OTHER_COLOR = "#9ca3af"
PHASE_COLORS = {
    "Retrieval": "#2a78d6",
    "Generation": "#eda100",
    "Evaluation": "#e34948",
    "Other": OTHER_COLOR,
}
STEP_COLORS = ["#7fb3e8", "#4f93df", "#2a78d6", "#1f5fae", "#174c8f", "#123a70"]

#: Bands thinner than this share of the drawing are widened to it, so a 0.1 ms
#: step is still visible. Labels and hovers always show the true time.
MIN_SHARE = 0.012
#: Retrieval is broken down further in the right-hand column, so its band gets
#: a larger minimum: otherwise its steps fan out as hairlines.
MIN_RETRIEVAL_SHARE = 0.04

# Retrieval steps per strategy: (label, telemetry key).
_STEPS = {
    "dense": [("Query embedding", "embed_query_ms"), ("Vector search", "vector_search_ms")],
    "hybrid": [("Query embedding", "embed_query_ms"), ("Dense search", "dense_search_ms"),
               ("BM25 search", "bm25_search_ms"), ("Rank fusion (RRF)", "rrf_fusion_ms")],
    "graph": [("Query embedding", "embed_query_ms"), ("Dense search", "dense_search_ms"),
              ("Entity seeding", "graph_seed_ms"), ("Graph traversal", "graph_traverse_ms"),
              ("Chunk scoring", "graph_score_ms"), ("Rank fusion (RRF)", "rrf_fusion_ms")],
}


def _is_dark_theme() -> bool:
    """Best-effort detection of the viewer's active Streamlit theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _fmt_pct(share: float) -> str:
    pct = 100 * share
    if pct == 0:
        return "0%"
    return "<0.1%" if pct < 0.1 else f"{pct:.1f}%"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _floored(values: list[float], total: float) -> list[float]:
    """Display widths: each at least MIN_SHARE of ``total``."""
    floor = MIN_SHARE * total
    return [max(v, floor) for v in values]


def _stack(heights: list[float], gap: float = 0.06, top: float = 0.04,
           bottom: float = 0.04) -> list[float]:
    """Centres (0 = top, 1 = bottom) for nodes stacked in one column."""
    usable = 1 - top - bottom - gap * (len(heights) - 1)
    total = sum(heights) or 1.0
    y, out = top, []
    for h in heights:
        span = usable * h / total
        out.append(y + span / 2)
        y += span + gap
    return out


def pipeline_flow_figure(
    retrieval_telemetry: dict,
    generation_ms: float,
    evaluation_ms: float,
    e2e_ms: float,
    ran_evaluation: bool,
    *,
    dark: bool = False,
) -> go.Figure:
    """Build the pipeline flow Sankey for one query (see the module docstring)."""
    strategy = retrieval_telemetry.get("strategy", "dense")
    steps = [(label, float(retrieval_telemetry.get(key, 0.0) or 0.0))
             for label, key in _STEPS.get(strategy, _STEPS["dense"])]
    retrieval_ms = sum(v for _, v in steps)

    phases = [("Retrieval", retrieval_ms), ("Generation", float(generation_ms or 0.0))]
    if ran_evaluation:
        phases.append(("Evaluation", float(evaluation_ms or 0.0)))
    # Whatever e2e_ms does not attribute to a named stage (token counting,
    # writing the telemetry record, Python glue) is shown, not dropped, so the
    # phases always add up to the total.
    other = max(float(e2e_ms or 0.0) - sum(v for _, v in phases), 0.0)
    phases.append(("Other", other))
    total = float(e2e_ms) if e2e_ms and e2e_ms > 0 else max(sum(v for _, v in phases), 1.0)

    # Display widths. Phases are floored against the total; retrieval's steps
    # are floored against retrieval's own width, then scaled to fill it, so
    # the bands leaving the Retrieval node match the band entering it.
    phase_w = _floored([v for _, v in phases], total)
    phase_w[0] = max(phase_w[0], MIN_RETRIEVAL_SHARE * total)
    retr_w = phase_w[0]
    raw = [max(v, MIN_SHARE * retr_w * 4) for _, v in steps]
    step_w = [retr_w * r / sum(raw) for r in raw] if sum(raw) else [retr_w / len(steps)] * len(steps)

    text = "#e8e8e3" if dark else "#1f2328"
    muted = "rgba(232,232,227,0.62)" if dark else "rgba(31,35,40,0.6)"

    def label(name: str, ms: float, share: float | None) -> str:
        detail = format_duration(ms) + ("" if share is None else f"  ·  {_fmt_pct(share)}")
        return f"<b>{name}</b><br><span style='color:{muted}'>{detail}</span>"

    names = {"Retrieval": f"Retrieval ({strategy})", "Generation": "Generation (LLM)",
             "Evaluation": "Evaluation (RAGAS)", "Other": "Other overhead"}

    labels = [label("End to end", total, None)]
    colors = [TOTAL_COLOR]
    xs, ys = [0.001], [0.5]
    hovers = [f"<b>End to end</b><br>{format_duration(total)}"]

    phase_y = _stack(phase_w)
    for (name, ms), y in zip(phases, phase_y, strict=True):
        labels.append(label(names[name], ms, ms / total))
        colors.append(PHASE_COLORS[name])
        xs.append(0.40)
        ys.append(y)
        hovers.append(f"<b>{names[name]}</b><br>{format_duration(ms)}, {_fmt_pct(ms / total)} of the query")

    # Retrieval's steps are spread evenly down the right column rather than
    # stacked inside retrieval's sliver of height, which is what keeps their
    # labels apart. Their bands fan out from the Retrieval node.
    first_step = len(labels)
    spacing = min(0.22, 0.88 / max(len(steps) - 1, 1))
    step_y = [0.06 + i * spacing for i in range(len(steps))]
    for i, ((name, ms), y) in enumerate(zip(steps, step_y, strict=True)):
        labels.append(label(name, ms, ms / total))
        colors.append(STEP_COLORS[min(i, len(STEP_COLORS) - 1)])
        xs.append(0.999)
        ys.append(y)
        share_r = ms / retrieval_ms if retrieval_ms else 0.0
        hovers.append(f"<b>{name}</b><br>{format_duration(ms)}, {_fmt_pct(share_r)} of retrieval, "
                      f"{_fmt_pct(ms / total)} of the query")

    sources, targets, values, link_colors, link_hover = [], [], [], [], []
    for i, ((name, ms), w) in enumerate(zip(phases, phase_w, strict=True), start=1):
        sources.append(0)
        targets.append(i)
        values.append(w)
        link_colors.append(_rgba(PHASE_COLORS[name], 0.32))
        link_hover.append(f"{names[name]}: {format_duration(ms)} ({_fmt_pct(ms / total)})")
    for j, ((name, ms), w) in enumerate(zip(steps, step_w, strict=True)):
        sources.append(1)
        targets.append(first_step + j)
        values.append(w)
        link_colors.append(_rgba(STEP_COLORS[min(j, len(STEP_COLORS) - 1)], 0.38))
        link_hover.append(f"{name}: {format_duration(ms)}")

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=labels, color=colors, x=xs, y=ys,
            pad=18, thickness=14, line=dict(width=0),
            customdata=hovers, hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            customdata=link_hover, hovertemplate="%{customdata}<extra></extra>",
        ),
        textfont=dict(size=14, color=text,
                      family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"),
    ))
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(color=text),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=13),
    )
    return fig


def pipeline_flow_sankey(
    retrieval_telemetry: dict,
    generation_ms: float,
    evaluation_ms: float,
    e2e_ms: float,
    ran_evaluation: bool,
) -> None:
    """Draw the pipeline flow Sankey for one query in Streamlit.

    ``ran_evaluation`` False omits the Evaluation phase entirely rather than
    showing a misleading near-zero band.
    """
    fig = pipeline_flow_figure(
        retrieval_telemetry, generation_ms, evaluation_ms, e2e_ms, ran_evaluation,
        dark=_is_dark_theme(),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
