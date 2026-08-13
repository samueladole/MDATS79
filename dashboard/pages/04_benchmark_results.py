"""
RAGScope Dashboard — Page 4: Benchmark Results
================================================
Visualises the results of the 200-query 2×2 factorial experiment.
Loads from the CSV output of ``experiments/run_2x2_factorial.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluation.hallucination_score import risk_band
from evaluation.metrics import cohens_d, pearson_r

st.title(":material/analytics: Benchmark Results")
st.caption(
    "Results of the 200-query, 2×2 factorial experiment "
    "(Llama3 vs Mistral × Dense vs Hybrid retrieval)."
)

# ── Load results CSV ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
csv_files = sorted(RESULTS_DIR.glob("benchmark_*.csv"), reverse=True) if RESULTS_DIR.exists() else []

if not csv_files:
    st.info(
        "No benchmark results found. Run the experiment first:\n\n"
        "```bash\n"
        "uv run python experiments/run_2x2_factorial.py --dataset all --conditions all\n"
        "```"
    )
    st.stop()

selected_file = st.selectbox(
    "Select results file",
    options=[f.name for f in csv_files],
    index=0,
)
df = pd.read_csv(RESULTS_DIR / selected_file)
st.markdown(f"Loaded **{len(df):,}** records from `{selected_file}`.")
st.divider()

# ── Condition identity ─────────────────────────────────────────────────────────
# Fixed colour per condition (never re-cycled), so a given condition is the same
# colour in every chart on this page regardless of which subset is loaded.
df["condition"] = (df["llm_model"].astype(str) + " + " + df["retrieval_strategy"].astype(str)).str.title()

CONDITION_ORDER = ["Llama3 + Dense", "Llama3 + Hybrid", "Mistral + Dense", "Mistral + Hybrid"]
CONDITION_COLORS = dict(zip(CONDITION_ORDER, ["#2a78d6", "#008300", "#e87ba4", "#eda100"]))
conditions_present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "UNKNOWN": "#95a5a6"}

TRANSPARENT_LAYOUT = {"plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"}

METRICS = [
    "context_relevance",
    "answer_faithfulness",
    "answer_correctness",
    "hallucination_risk",
    "e2e_ms",
    "total_tokens",
]
QUALITY_METRICS = ["context_relevance", "answer_faithfulness", "answer_correctness"]

# ── Summary statistics ────────────────────────────────────────────────────────
st.subheader("Summary Statistics by Condition")
st.caption("Mean, standard deviation, and median for each metric, grouped by experimental condition.")

agg = df.groupby("condition")[METRICS].agg(["mean", "std", "median"]).round(3)
st.dataframe(agg, width="stretch")

# Grouped bar: mean RAGAS scores by condition, alongside the table above.
quality_mean = df.groupby("condition")[QUALITY_METRICS].mean().reset_index()
quality_std = df.groupby("condition")[QUALITY_METRICS].std().reset_index()
quality_long = quality_mean.melt(id_vars="condition", var_name="metric", value_name="mean")
std_long = quality_std.melt(id_vars="condition", var_name="metric", value_name="std")
quality_long = quality_long.merge(std_long, on=["condition", "metric"])
quality_long["metric"] = quality_long["metric"].str.replace("_", " ").str.title()

fig_bar = px.bar(
    quality_long,
    x="metric",
    y="mean",
    color="condition",
    error_y="std",
    barmode="group",
    category_orders={"condition": conditions_present},
    color_discrete_map=CONDITION_COLORS,
    labels={"metric": "RAGAS Metric", "mean": "Mean Score", "condition": "Condition"},
    title="Mean RAGAS Scores by Condition (error bars = ±1 SD)",
)
fig_bar.update_layout(height=420, yaxis_range=[0, 1.05], **TRANSPARENT_LAYOUT)
st.plotly_chart(fig_bar, width="stretch")
st.divider()

# ── Score distributions ───────────────────────────────────────────────────────
st.subheader("Score Distributions")
st.caption(
    "Box shows the interquartile range and median for the selected metric; whiskers "
    "extend to the min/max excluding outliers, dots are individual outlier queries."
)
metric_choice = st.selectbox(
    "Metric",
    options=METRICS,
    format_func=lambda x: x.replace("_", " ").title(),
)

fig = px.box(
    df.dropna(subset=[metric_choice]),
    x="condition",
    y=metric_choice,
    color="condition",
    category_orders={"condition": conditions_present},
    color_discrete_map=CONDITION_COLORS,
    points="outliers",
    labels={"condition": "Condition", metric_choice: metric_choice.replace("_", " ").title()},
    title=f"{metric_choice.replace('_', ' ').title()} Distribution by Condition",
)
fig.update_layout(height=420, showlegend=False, **TRANSPARENT_LAYOUT)
st.plotly_chart(fig, width="stretch")
st.divider()

# ── Hallucination risk profile ────────────────────────────────────────────────
st.subheader("Hallucination Risk Profile")
if "hallucination_risk" in df.columns:
    risk_df = df.dropna(subset=["hallucination_risk"]).copy()
    risk_df["risk_band"] = risk_df["hallucination_risk"].apply(risk_band)

    band_counts = risk_df.groupby(["condition", "risk_band"]).size().reset_index(name="count")
    totals = risk_df.groupby("condition").size().reset_index(name="total")
    band_counts = band_counts.merge(totals, on="condition")
    band_counts["pct"] = (band_counts["count"] / band_counts["total"] * 100).round(1)

    fig_risk = px.bar(
        band_counts,
        x="condition",
        y="pct",
        color="risk_band",
        category_orders={"condition": conditions_present, "risk_band": RISK_ORDER},
        color_discrete_map=RISK_COLORS,
        barmode="stack",
        text="count",
        labels={"condition": "Condition", "pct": "Share of Queries (%)", "risk_band": "Risk Band"},
        title="Hallucination Risk Band Distribution by Condition",
    )
    fig_risk.update_traces(texttemplate="%{text}", textposition="inside")
    fig_risk.update_layout(height=420, yaxis_range=[0, 100], **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig_risk, width="stretch")
    st.caption("Bar segments show query counts; bar height is each band's share of that condition's queries.")
st.divider()

# ── Effect sizes (Cohen's d) ──────────────────────────────────────────────────
st.subheader("Effect Sizes (Cohen's d)")
st.caption("|d| < 0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · > 0.8 large (Cohen, 1988)")

COHENS_D_COMPARISONS = [
    ("llama3", "dense", "mistral", "dense", "LLM effect (Dense)"),
    ("llama3", "hybrid", "mistral", "hybrid", "LLM effect (Hybrid)"),
    ("llama3", "dense", "llama3", "hybrid", "Retrieval effect (Llama3)"),
    ("mistral", "dense", "mistral", "hybrid", "Retrieval effect (Mistral)"),
]
COMPARISON_COLORS = dict(
    zip([c[4] for c in COHENS_D_COMPARISONS], ["#2a78d6", "#008300", "#e87ba4", "#eda100"])
)
EFFECT_METRICS = [*QUALITY_METRICS, "hallucination_risk"]

effect_rows = []
for llm_a, ret_a, llm_b, ret_b, label in COHENS_D_COMPARISONS:
    group_a = df[(df["llm_model"] == llm_a) & (df["retrieval_strategy"] == ret_a)]
    group_b = df[(df["llm_model"] == llm_b) & (df["retrieval_strategy"] == ret_b)]
    if len(group_a) < 2 or len(group_b) < 2:
        continue
    for metric in EFFECT_METRICS:
        if metric not in df.columns:
            continue
        result = cohens_d(
            group_a[metric].dropna().tolist(),
            group_b[metric].dropna().tolist(),
            label_a=f"{llm_a}+{ret_a}",
            label_b=f"{llm_b}+{ret_b}",
            metric=metric,
        )
        effect_rows.append(
            {
                "comparison": label,
                "metric": metric.replace("_", " ").title(),
                "d": result.d,
                "magnitude": result.magnitude,
                "mean_a": result.mean_a,
                "mean_b": result.mean_b,
            }
        )

if effect_rows:
    effect_df = pd.DataFrame(effect_rows)
    fig_d = px.bar(
        effect_df,
        x="d",
        y="metric",
        color="comparison",
        orientation="h",
        barmode="group",
        color_discrete_map=COMPARISON_COLORS,
        hover_data=["magnitude", "mean_a", "mean_b"],
        labels={"d": "Cohen's d", "metric": "Metric", "comparison": "Comparison"},
        title="Effect Sizes — LLM vs Retrieval Strategy",
    )
    for threshold in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        fig_d.add_vline(x=threshold, line_dash="dot", line_color="rgba(128,128,128,0.4)")
    fig_d.add_vline(x=0, line_color="rgba(128,128,128,0.8)")
    fig_d.update_layout(height=460, **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig_d, width="stretch")

    with st.expander("Effect size detail table", icon=":material/straighten:"):
        st.dataframe(effect_df.round(4), width="stretch")
else:
    st.info("Effect-size comparisons need at least two conditions with ≥2 samples each in the loaded results.")
st.divider()

# ── Latency & token cost breakdown ────────────────────────────────────────────
st.subheader("Latency & Token Cost Breakdown")
st.caption(
    "Left: mean end-to-end latency split into retrieval vs. generation time per condition. "
    "Right: mean prompt vs. completion token counts per condition."
)

col_a, col_b = st.columns(2)

with col_a:
    latency_means = df.groupby("condition")[["retrieval_ms", "generation_ms"]].mean().reset_index()
    latency_long = latency_means.melt(id_vars="condition", var_name="stage", value_name="ms")
    latency_long["stage"] = latency_long["stage"].map(
        {"retrieval_ms": "Retrieval", "generation_ms": "Generation"}
    )
    fig_lat = px.bar(
        latency_long,
        x="condition",
        y="ms",
        color="stage",
        category_orders={"condition": conditions_present},
        color_discrete_map={"Retrieval": "#2a78d6", "Generation": "#eda100"},
        barmode="stack",
        labels={"condition": "Condition", "ms": "Mean Latency (ms)", "stage": "Stage"},
        title="Mean Latency Breakdown by Condition",
    )
    fig_lat.update_layout(height=380, **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig_lat, width="stretch")

with col_b:
    token_means = df.groupby("condition")[["prompt_tokens", "completion_tokens"]].mean().reset_index()
    token_long = token_means.melt(id_vars="condition", var_name="token_type", value_name="tokens")
    token_long["token_type"] = token_long["token_type"].map(
        {"prompt_tokens": "Prompt", "completion_tokens": "Completion"}
    )
    fig_tok = px.bar(
        token_long,
        x="condition",
        y="tokens",
        color="token_type",
        category_orders={"condition": conditions_present},
        color_discrete_map={"Prompt": "#4a3aa7", "Completion": "#1baf7a"},
        barmode="stack",
        labels={"condition": "Condition", "tokens": "Mean Token Count", "token_type": "Token Type"},
        title="Mean Token Usage by Condition",
    )
    fig_tok.update_layout(height=380, **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig_tok, width="stretch")

if "answer_faithfulness" in df.columns and "e2e_ms" in df.columns:
    st.markdown("**Latency vs. Answer Faithfulness**")
    fig2 = px.scatter(
        df.dropna(subset=["answer_faithfulness", "e2e_ms"]),
        x="e2e_ms",
        y="answer_faithfulness",
        color="condition",
        category_orders={"condition": conditions_present},
        color_discrete_map=CONDITION_COLORS,
        opacity=0.6,
        trendline="ols",
        labels={"e2e_ms": "End-to-End Latency (ms)", "answer_faithfulness": "Answer Faithfulness"},
        title="Latency vs. Faithfulness (all conditions)",
    )
    fig2.update_layout(height=680, **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig2, width="stretch")
st.divider()

# ── Metric correlations ───────────────────────────────────────────────────────
st.subheader("Metric Correlations")
st.caption(
    "Pearson r across all recorded metrics. Constant columns (e.g. cost, when running "
    "local inference) and pairs with fewer than 3 valid values are omitted."
)

CORRELATION_METRICS = [
    "context_relevance",
    "answer_faithfulness",
    "answer_correctness",
    "hallucination_risk",
    "retrieval_ms",
    "generation_ms",
    "e2e_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "chunks_retrieved",
    "top_chunk_score",
]
CORRELATION_LABELS = {
    "context_relevance": "Context Relevance",
    "answer_faithfulness": "Answer Faithfulness",
    "answer_correctness": "Answer Correctness",
    "hallucination_risk": "Hallucination Risk",
    "retrieval_ms": "Retrieval Latency (ms)",
    "generation_ms": "Generation Latency (ms)",
    "e2e_ms": "E2E Latency (ms)",
    "prompt_tokens": "Prompt Tokens",
    "completion_tokens": "Completion Tokens",
    "total_tokens": "Total Tokens",
    "chunks_retrieved": "Chunks Retrieved",
    "top_chunk_score": "Top Chunk Score",
}

corr_metrics = [m for m in CORRELATION_METRICS if m in df.columns and df[m].dropna().std() > 0]

if len(corr_metrics) >= 2:
    corr_matrix = [
        [pearson_r(df[row_metric].tolist(), df[col_metric].tolist()) for col_metric in corr_metrics]
        for row_metric in corr_metrics
    ]
    labels = [CORRELATION_LABELS.get(m, m.replace("_", " ").title()) for m in corr_metrics]
    z = [[v if v is not None else 0.0 for v in row] for row in corr_matrix]
    text = [[f"{v:.2f}" if v is not None else "—" for v in row] for row in corr_matrix]

    fig_corr = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} × %{x}<br>Pearson r = %{z:.3f}<extra></extra>",
            colorbar=dict(title="r"),
        )
    )
    fig_corr.update_layout(
        title="Pearson Correlation — All Metrics",
        height=max(820, 40 * len(labels)),
        margin=dict(l=0, r=0, t=50, b=100),
        xaxis=dict(side="top", tickangle=-45),
        **TRANSPARENT_LAYOUT,
    )
    st.plotly_chart(fig_corr, width="stretch")
else:
    st.info("Not enough varying numeric metrics in this file to compute correlations.")
st.divider()

# ── Per-dataset breakdown ─────────────────────────────────────────────────────
if "dataset" in df.columns:
    st.subheader("Performance by Dataset")
    st.caption(
        "Mean quality scores per BioASQ query role (Phase A retrieval / factoid / summary) "
        "× condition — useful for spotting whether a condition struggles on a specific "
        "query type (e.g. summary questions requiring multi-passage synthesis)."
    )
    dataset_agg = (
        df.groupby(["dataset", "condition"])[
            ["context_relevance", "answer_faithfulness", "hallucination_risk"]
        ]
        .mean()
        .round(3)
        .reset_index()
    )
    datasets_present = list(dataset_agg["dataset"].unique())

    dataset_metric_choice = st.selectbox(
        "Metric to chart",
        options=["context_relevance", "answer_faithfulness", "hallucination_risk"],
        format_func=lambda x: x.replace("_", " ").title(),
        key="dataset_metric_choice",
    )

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        fig_dataset = px.bar(
            dataset_agg,
            x="dataset",
            y=dataset_metric_choice,
            color="condition",
            barmode="group",
            category_orders={"condition": conditions_present, "dataset": datasets_present},
            color_discrete_map=CONDITION_COLORS,
            labels={
                "dataset": "Dataset",
                dataset_metric_choice: dataset_metric_choice.replace("_", " ").title(),
                "condition": "Condition",
            },
            title=f"{dataset_metric_choice.replace('_', ' ').title()} by Dataset × Condition",
        )
        fig_dataset.update_layout(height=420, yaxis_range=[0, 1.05], **TRANSPARENT_LAYOUT)
        st.plotly_chart(fig_dataset, width="stretch")
    with col_table:
        st.dataframe(dataset_agg, width="stretch", hide_index=True)
    st.divider()

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Full Results Table", icon=":material/table_chart:"):
    st.dataframe(df, width="stretch")
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        icon=":material/download:",
        data=csv,
        file_name=selected_file,
        mime="text/csv",
    )
