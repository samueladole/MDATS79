"""
RAGScope — Telemetry Logger
=============================
Records a complete, structured JSON log entry for every query execution.
Each entry captures the full pipeline trace:
    * Query metadata (text, dataset, query type)
    * Retrieval telemetry (strategy, latency, chunk scores)
    * Generation telemetry (model, latency, token counts, cost)
    * Evaluation scores (RAGAS metrics, hallucination risk)
    * Timestamps

Log entries are written to individual JSON files under
``settings.telemetry_store_dir`` and are the primary data source for:
    - The Streamlit dashboard (live monitor and query explorer)
    - The benchmark experiment analysis (Phase 8)

Schema is intentionally flat to simplify pandas ingestion for analysis.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import orjson
from loguru import logger as _logger

from config.settings import settings
from pipeline.vectorstore import RetrievedChunk
from telemetry.token_counter import TokenUsage

# ── Telemetry record ─────────────────────────────────────────────────────────


def build_record(
    *,
    query_id: str | None,
    query: str,
    dataset: str,
    query_type: str,
    llm_model: str,
    retrieval_strategy: str,
    # Retrieval telemetry
    retrieved_chunks: list[RetrievedChunk],
    retrieval_ms: float,
    embed_query_ms: float,
    # Generation telemetry
    answer: str,
    generation_ms: float,
    token_usage: TokenUsage,
    # Evaluation scores (populated after RAGAS runs)
    context_relevance: float | None = None,
    answer_faithfulness: float | None = None,
    answer_correctness: float | None = None,
    hallucination_risk: float | None = None,
    # Ground truth (for answer_correctness computation)
    ground_truth: str | None = None,
    # Experiment metadata
    experiment_id: str | None = None,
    condition_label: str | None = None,
) -> dict:
    """
    Assemble a complete telemetry record dict.

    All numeric fields are rounded to 4 decimal places to keep log
    files compact while preserving sufficient precision for analysis.

    Returns
    -------
    dict
        Flat JSON-serialisable record. Field names are snake_case to
        simplify downstream pandas ``pd.read_json`` / ``pd.DataFrame`` usage.
    """
    now = datetime.now(UTC)
    e2e_ms = round(retrieval_ms + generation_ms, 2)

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "record_id": query_id or str(uuid.uuid4()),
        "timestamp_utc": now.isoformat(),
        "experiment_id": experiment_id or "",
        "condition_label": condition_label or f"{llm_model}_{retrieval_strategy}",
        # ── Query ─────────────────────────────────────────────────────────────
        "query": query,
        "dataset": dataset,
        "query_type": query_type,
        "ground_truth": ground_truth or "",
        # ── Model config ──────────────────────────────────────────────────────
        "llm_model": llm_model,
        "retrieval_strategy": retrieval_strategy,
        # ── Retrieval ─────────────────────────────────────────────────────────
        "chunks_retrieved": len(retrieved_chunks),
        "top_chunk_score": round(retrieved_chunks[0].score, 4) if retrieved_chunks else 0.0,
        "mean_chunk_score": round(sum(c.score for c in retrieved_chunks) / len(retrieved_chunks), 4)
        if retrieved_chunks
        else 0.0,
        "retrieved_chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text[:500],  # truncated for log compactness
                "score": round(c.score, 4),
                "dataset": c.metadata.get("dataset", ""),
            }
            for c in retrieved_chunks
        ],
        "embed_query_ms": round(embed_query_ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        # ── Generation ────────────────────────────────────────────────────────
        "answer": answer,
        "generation_ms": round(generation_ms, 2),
        "e2e_ms": e2e_ms,
        "prompt_tokens": token_usage.prompt_tokens,
        "completion_tokens": token_usage.completion_tokens,
        "total_tokens": token_usage.total_tokens,
        "estimated_cost_usd": token_usage.estimated_cost_usd,
        # ── Evaluation ────────────────────────────────────────────────────────
        "context_relevance": round(context_relevance, 4) if context_relevance is not None else None,
        "answer_faithfulness": round(answer_faithfulness, 4)
        if answer_faithfulness is not None
        else None,
        "answer_correctness": round(answer_correctness, 4)
        if answer_correctness is not None
        else None,
        "hallucination_risk": round(hallucination_risk, 4)
        if hallucination_risk is not None
        else None,
    }


# ── Logger class ──────────────────────────────────────────────────────────────


class TelemetryLogger:
    """
    Writes telemetry records to the configured store directory.

    Each record is written as an individual ``<record_id>.json`` file.
    This design avoids write contention in concurrent experiment runs
    and makes individual record inspection straightforward.

    Parameters
    ----------
    store_dir : Directory for JSON files. Defaults to ``settings.telemetry_store_dir``.
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or settings.telemetry_store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        _logger.setLevel(settings.log_level)

    def log(self, record: dict) -> Path:
        """
        Persist a telemetry record to disk.

        Parameters
        ----------
        record : Dict produced by ``build_record()``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        record_id = record.get("record_id", str(uuid.uuid4()))
        path = self.store_dir / f"{record_id}.json"
        path.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
        _logger.debug(f"Telemetry written → {path.name}")
        return path

    def load_all(self, limit: int | None = None) -> list[dict]:
        """
        Load all telemetry records from the store directory.

        Parameters
        ----------
        limit : Maximum number of records to return (most recent first).
                Defaults to ``settings.telemetry_max_records``.

        Returns
        -------
        list[dict]
            Records sorted by ``timestamp_utc`` descending.
        """
        limit = limit or settings.telemetry_max_records
        files = sorted(
            self.store_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        records = []
        for f in files:
            try:
                records.append(orjson.loads(f.read_bytes()))
            except Exception as exc:
                _logger.warning(f"Failed to load telemetry record {f.name}: {exc}")
        return records

    def count(self) -> int:
        """Return the number of telemetry records in the store."""
        return sum(1 for _ in self.store_dir.glob("*.json"))


# ── Module-level singleton ─────────────────────────────────────────────────────
_tel_logger: TelemetryLogger | None = None


def get_telemetry_logger() -> TelemetryLogger:
    global _tel_logger
    if _tel_logger is None:
        _tel_logger = TelemetryLogger()
    return _tel_logger
