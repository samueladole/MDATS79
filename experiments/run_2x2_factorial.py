"""
RAGScope — 2×2 Factorial Experiment Runner
============================================
Executes the full benchmark experiment across all four experimental conditions:

    Condition A: Llama3  + Dense retrieval
    Condition B: Llama3  + Hybrid retrieval
    Condition C: Mistral + Dense retrieval
    Condition D: Mistral + Hybrid retrieval

Each condition is evaluated against the full 200-query benchmark
(60 BioASQ Phase A + 40 BioASQ factoid + 100 BioASQ summary). Results
are written to a timestamped CSV file in ``experiments/results/``.

Expected runtime: 4–8 hours on CPU (consumer hardware).
Run as an overnight job. Use ``--resume`` to continue from a checkpoint.

Usage
-----
    # Full experiment (all 4 conditions)
    uv run python experiments/run_2x2_factorial.py --dataset all --conditions all

    # Single condition
    uv run python experiments/run_2x2_factorial.py \\
        --llm llama3 --retrieval hybrid --dataset all

    # Resume after interruption
    uv run python experiments/run_2x2_factorial.py --dataset all --conditions all --resume
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from evaluation.benchmark import load_benchmark, run_benchmark
from pipeline.rag import RAGPipeline

app = typer.Typer(add_completion=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# All 4 conditions in the 2×2 factorial design
ALL_CONDITIONS: list[tuple[str, str]] = [
    ("llama3", "dense"),
    ("llama3", "hybrid"),
    ("mistral", "dense"),
    ("mistral", "hybrid"),
]


@app.command()
def main(
    dataset: str = typer.Option(
        "all", help="Dataset subset: all | bioasq_phase_a | bioasq_factoid | bioasq_summary"
    ),
    conditions: str = typer.Option(
        "all",
        help="Conditions to run: 'all' or comma-separated e.g. 'llama3_dense,mistral_hybrid'",
    ),
    llm: str = typer.Option(
        "", help="LLM for single-condition run: llama3 | mistral (overrides --conditions)"
    ),
    retrieval: str = typer.Option(
        "", help="Retrieval for single-condition run: dense | hybrid (overrides --conditions)"
    ),
    output: str = typer.Option(
        "",
        help="Output CSV path. Defaults to experiments/results/benchmark_<timestamp>.csv",
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from checkpoint if a previous run was interrupted."
    ),
    seed: int = typer.Option(42, help="Random seed for query sampling."),
) -> None:
    """Run the 2×2 factorial RAG benchmark experiment."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve conditions ─────────────────────────────────────────────────────
    if llm and retrieval:
        selected_conditions = [(llm, retrieval)]
    elif conditions == "all":
        selected_conditions = ALL_CONDITIONS
    else:
        selected_conditions = []
        for token in conditions.split(","):
            parts = token.strip().split("_")
            if len(parts) == 2:
                selected_conditions.append((parts[0], parts[1]))

    if not selected_conditions:
        logger.error("No valid conditions specified. Aborting.")
        raise typer.Exit(1)

    logger.info(
        f"Experiment: {len(selected_conditions)} condition(s), dataset={dataset}, seed={seed}"
    )

    # ── Load benchmark queries (once, reused across conditions) ────────────────
    benchmark_queries = load_benchmark(seed=seed)

    # Optionally filter to a single dataset
    if dataset != "all":
        benchmark_queries = [q for q in benchmark_queries if q.get("dataset") == dataset]
        logger.info(f"Filtered to {dataset}: {len(benchmark_queries)} queries.")

    # ── Run each condition ─────────────────────────────────────────────────────
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"
    all_results: list[dict] = []

    for llm_model, ret_strategy in selected_conditions:
        condition_label = f"{llm_model}_{ret_strategy}"
        logger.info(f"─── Condition: {condition_label} ───")

        checkpoint_path = RESULTS_DIR / f"checkpoint_{experiment_id}_{condition_label}.json"

        pipeline = RAGPipeline(
            llm_model=llm_model,
            retrieval_strategy=ret_strategy,
            run_evaluation=True,
            experiment_id=experiment_id,
        )

        condition_results = run_benchmark(
            pipeline=pipeline,
            queries=benchmark_queries,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

        # Tag each record with condition metadata
        for r in condition_results:
            r["llm_model"] = llm_model
            r["retrieval_strategy"] = ret_strategy
            r["experiment_id"] = experiment_id

        all_results.extend(condition_results)
        logger.info(f"Condition {condition_label} complete: {len(condition_results)} records.")

        # Clean up checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    # ── Save results ───────────────────────────────────────────────────────────
    if not output:
        output = str(RESULTS_DIR / f"benchmark_{timestamp}.csv")

    df = pd.DataFrame(all_results)
    df.to_csv(output, index=False)

    logger.info(f"Results saved → {output}")
    logger.info(
        f"Experiment complete: {len(all_results)} total records "
        f"across {len(selected_conditions)} condition(s)."
    )

    # ── Quick summary ──────────────────────────────────────────────────────────
    if "context_relevance" in df.columns:
        summary = (
            df.groupby(["llm_model", "retrieval_strategy"])[
                [
                    "context_relevance",
                    "answer_faithfulness",
                    "hallucination_risk",
                    "e2e_ms",
                ]
            ]
            .mean()
            .round(3)
        )
        logger.info(f"\n{summary.to_string()}")


if __name__ == "__main__":
    app()
