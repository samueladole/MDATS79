"""
RAGScope — Statistical Analysis Script
========================================
Loads the benchmark experiment CSV and produces a structured
Markdown analysis report including:
    * Descriptive statistics per condition and metric
    * Cohen's d effect sizes for all pairwise condition comparisons
    * Pearson correlations between latency and quality metrics
    * Per-dataset performance breakdown

Usage
-----
    uv run python experiments/analyse_results.py \\
        --input experiments/results/benchmark_20260801.csv \\
        --output experiments/results/analysis_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from evaluation.metrics import (
    EVAL_METRICS,
    cohens_d,
    descriptive_stats,
    pearson_r,
)

app = typer.Typer(add_completion=False)

CONDITIONS = [
    ("llama3", "dense", "Llama3+Dense"),
    ("llama3", "hybrid", "Llama3+Hybrid"),
    ("qwen", "dense", "Qwen+Dense"),
    ("qwen", "hybrid", "Qwen+Hybrid"),
]


@app.command()
def main(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to benchmark CSV."),
    output_path: str = typer.Option(
        "", "--output", "-o", help="Output Markdown report path. Defaults to <input>_report.md"
    ),
) -> None:
    """Generate statistical analysis report from benchmark results CSV."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    csv_path = Path(input_path)
    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        raise typer.Exit(1)

    if not output_path:
        output_path = str(csv_path.with_suffix("")) + "_report.md"

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} records from {csv_path.name}.")

    lines: list[str] = _build_report(df, csv_path.name)

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written → {output_path}")


def _build_report(df: pd.DataFrame, source_file: str) -> list[str]:
    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────────────
    lines += [
        "# RAGScope — Benchmark Analysis Report",
        "",
        f"**Source file:** `{source_file}`  ",
        f"**Total records:** {len(df):,}  ",
        f"**Conditions:** {df['llm_model'].nunique() if 'llm_model' in df.columns else '?'} LLMs "
        f"× {df['retrieval_strategy'].nunique() if 'retrieval_strategy' in df.columns else '?'} retrieval strategies  ",
        "",
        "---",
        "",
    ]

    # ── Descriptive statistics per condition ────────────────────────────────────
    lines += ["## 1. Descriptive Statistics", ""]
    for llm, strategy, label in CONDITIONS:
        subset_df = df[
            (df.get("llm_model", pd.Series()) == llm)
            & (df.get("retrieval_strategy", pd.Series()) == strategy)
        ]
        if len(subset_df) == 0:
            continue
        lines.append(f"### {label}  (n={len(subset_df)})")
        lines.append("")
        lines.append("| Metric | Mean | Std | Median | IQR | Min | Max |")
        lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for metric in EVAL_METRICS:
            vals = subset_df[metric].dropna().tolist() if metric in subset_df.columns else []
            s = descriptive_stats(vals, metric)
            if s.n == 0:
                continue
            lines.append(
                f"| {metric.replace('_', ' ').title()} "
                f"| {s.mean:.4f} | {s.std:.4f} | {s.median:.4f} "
                f"| {s.iqr:.4f} | {s.minimum:.4f} | {s.maximum:.4f} |"
            )
        lines.append("")

    # ── Cohen's d pairwise comparisons ─────────────────────────────────────────
    lines += [
        "## 2. Effect Sizes (Cohen's d)",
        "",
        "_Interpretation: |d| < 0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · > 0.8 large_",
        "",
    ]

    pairs = [
        ("llama3", "dense", "llama3", "hybrid", "LLM: Llama3 — Dense vs Hybrid"),
        ("qwen", "dense", "qwen", "hybrid", "LLM: Qwen — Dense vs Hybrid"),
        ("llama3", "dense", "qwen", "dense", "Retrieval: Dense — Llama3 vs Qwen"),
        ("llama3", "hybrid", "qwen", "hybrid", "Retrieval: Hybrid — Llama3 vs Qwen"),
    ]
    for llm_a, ret_a, llm_b, ret_b, comparison in pairs:
        a_df = df[(df.get("llm_model", "") == llm_a) & (df.get("retrieval_strategy", "") == ret_a)]
        b_df = df[(df.get("llm_model", "") == llm_b) & (df.get("retrieval_strategy", "") == ret_b)]
        lines += [
            f"### {comparison}",
            "",
            "| Metric | Cohen's d | Magnitude | Mean A | Mean B |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
        for metric in [
            "context_relevance",
            "answer_faithfulness",
            "answer_correctness",
            "hallucination_risk",
            "e2e_ms",
        ]:
            if metric not in df.columns:
                continue
            result = cohens_d(
                a_df[metric].dropna().tolist(),
                b_df[metric].dropna().tolist(),
                label_a=f"{llm_a}+{ret_a}",
                label_b=f"{llm_b}+{ret_b}",
                metric=metric,
            )
            lines.append(
                f"| {metric.replace('_', ' ').title()} "
                f"| {result.d:+.4f} | {result.magnitude} "
                f"| {result.mean_a:.4f} | {result.mean_b:.4f} |"
            )
        lines.append("")

    # ── Pearson correlations ────────────────────────────────────────────────────
    lines += ["## 3. Pearson Correlations (Latency vs Quality)", ""]
    lines += ["| Metric Pair | r | Interpretation |", "|---|:---:|---|"]
    latency_metric = "e2e_ms"
    for qmetric in [
        "context_relevance",
        "answer_faithfulness",
        "answer_correctness",
        "hallucination_risk",
    ]:
        if qmetric not in df.columns or latency_metric not in df.columns:
            continue
        paired = df[[latency_metric, qmetric]].dropna()
        r = pearson_r(paired[latency_metric].tolist(), paired[qmetric].tolist())
        if r is None:
            continue
        interp = (
            "strong positive"
            if r > 0.7
            else "moderate positive"
            if r > 0.3
            else "weak positive"
            if r > 0
            else "weak negative"
            if r > -0.3
            else "moderate negative"
            if r > -0.7
            else "strong negative"
        )
        lines.append(
            f"| E2E Latency vs {qmetric.replace('_', ' ').title()} | {r:+.4f} | {interp} |"
        )
    lines.append("")

    # ── Dataset breakdown ───────────────────────────────────────────────────────
    if "dataset" in df.columns:
        lines += ["## 4. Performance by Dataset", ""]
        lines += [
            "| Dataset | Condition | Context Rel. | Faithfulness | Hall. Risk |",
            "|---|---|:---:|:---:|:---:|",
        ]
        for ds in df["dataset"].unique():
            for llm, ret, label in CONDITIONS:
                sub = df[
                    (df["dataset"] == ds)
                    & (df.get("llm_model", "") == llm)
                    & (df.get("retrieval_strategy", "") == ret)
                ]
                if len(sub) == 0:
                    continue
                lines.append(
                    f"| {ds} | {label} "
                    f"| {sub['context_relevance'].mean():.3f} "
                    f"| {sub['answer_faithfulness'].mean():.3f} "
                    f"| {sub['hallucination_risk'].mean():.3f} |"
                )
        lines.append("")

    lines += ["---", "", "_Generated by RAGScope analysis pipeline._"]
    return lines


if __name__ == "__main__":
    app()
