"""
RAGScope — Benchmark Runner
=============================
Loads the 200-query evaluation benchmark (60 MS MARCO + 40 NQ + 100 HotpotQA)
and executes queries through the RAG pipeline, collecting full telemetry
and RAGAS scores for each query.

This module is used by ``experiments/run_2x2_factorial.py`` to produce
the dataset for Phase 8 analysis.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from config.settings import settings
from data.loaders import hotpotqa as hq_loader
from data.loaders import msmarco as mm_loader
from data.loaders import natural_questions as nq_loader


def load_benchmark(seed: int | None = None) -> list[dict]:
    """
    Load the full 200-query evaluation benchmark.

    Each record has at minimum:
        ``id``, ``query``, ``answers``, ``query_type``, ``dataset``

    Returns
    -------
    list[dict]  Shuffled, 200-item benchmark query list.
    """
    import random

    seed = seed if seed is not None else settings.experiment_random_seed
    logger.info("Loading benchmark queries …")

    ms_queries = mm_loader.load_queries(sample_size=60, seed=seed)
    nq_queries, _ = nq_loader.load_queries_and_passages(sample_size=40, seed=seed)
    hq_queries = hq_loader.load_queries(sample_size=100, seed=seed)

    benchmark = ms_queries + nq_queries + hq_queries
    rng = random.Random(seed)
    rng.shuffle(benchmark)

    logger.info(
        f"Benchmark assembled: {len(ms_queries)} MS MARCO + "
        f"{len(nq_queries)} NQ + {len(hq_queries)} HotpotQA = "
        f"{len(benchmark)} total queries."
    )
    return benchmark


def run_benchmark(
    pipeline,
    queries: list[dict],
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> list[dict]:
    """
    Execute all benchmark queries through ``pipeline`` and collect results.

    Parameters
    ----------
    pipeline         : An initialised ``RAGPipeline`` instance.
    queries          : Query list from ``load_benchmark()``.
    checkpoint_path  : Optional path to persist completed record IDs.
                       Enables ``--resume`` functionality.
    resume           : If True, skip queries already in the checkpoint.

    Returns
    -------
    list[dict]
        List of telemetry record dicts, one per query.
    """
    completed_ids: set[str] = set()
    if resume and checkpoint_path and checkpoint_path.exists():
        import orjson

        completed_ids = set(orjson.loads(checkpoint_path.read_bytes()))
        logger.info(f"Resuming: {len(completed_ids)} queries already completed.")

    results: list[dict] = []
    failed: list[str] = []

    for q in tqdm(queries, desc="Benchmark", unit="query", file=sys.stdout):
        qid = q["id"]
        if qid in completed_ids:
            continue

        ground_truth = (q.get("answers") or [None])[0]

        try:
            result = pipeline.query(
                query_text=q["query"],
                ground_truth=ground_truth,
                query_id=qid,
                query_type=q.get("query_type", "unknown"),
            )

            record = {
                "query_id": qid,
                "query": q["query"],
                "dataset": q.get("dataset", ""),
                "query_type": q.get("query_type", ""),
                "ground_truth": ground_truth or "",
                "answer": result.answer,
                "context_relevance": result.context_relevance,
                "answer_faithfulness": result.answer_faithfulness,
                "answer_correctness": result.answer_correctness,
                "hallucination_risk": result.hallucination_risk,
                "retrieval_ms": result.retrieval_telemetry.get("retrieval_ms", 0),
                "generation_ms": result.generation_response.generation_ms,
                "e2e_ms": result.e2e_ms,
                "prompt_tokens": result.token_usage.prompt_tokens,
                "completion_tokens": result.token_usage.completion_tokens,
                "total_tokens": result.token_usage.total_tokens,
                "estimated_cost_usd": result.token_usage.estimated_cost_usd,
                "chunks_retrieved": len(result.retrieved_chunks),
                "top_chunk_score": result.retrieved_chunks[0].score
                if result.retrieved_chunks
                else 0.0,
                "llm_model": pipeline.llm_model,
                "retrieval_strategy": pipeline.retrieval_strategy,
            }
            results.append(record)
            completed_ids.add(qid)

            # Persist checkpoint after each query
            if checkpoint_path:
                import orjson

                checkpoint_path.write_bytes(orjson.dumps(list(completed_ids)))

        except Exception as exc:
            logger.error(f"Query {qid} failed: {exc}")
            failed.append(qid)
            time.sleep(1)  # brief pause before retrying next query

    if failed:
        logger.warning(f"{len(failed)} queries failed: {failed}")

    logger.info(f"Benchmark complete: {len(results)} results collected.")
    return results
