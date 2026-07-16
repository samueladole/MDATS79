"""
RAGScope — RAG Query Pipeline
===============================
The central orchestrator for a single end-to-end RAG query execution.

Execution flow
--------------
  1. Retrieve top-k chunks (dense or hybrid)
  2. Inject retrieved context into the LLM prompt
  3. Generate a response via Ollama (Llama 3 or Mistral)
  4. Count tokens and estimate cost
  5. Run RAGAS evaluation metrics
  6. Compute hallucination risk score
  7. Build and persist a telemetry record
  8. Return a structured ``RAGResult``

Every module in the pipeline is called through this orchestrator.
The Streamlit dashboard and the benchmark experiment runner both
use this class as their single entry point.

Usage (CLI)
-----------
    uv run python pipeline/rag.py \\
        --query "What causes hallucination in LLMs?" \\
        --llm llama3 \\
        --retrieval hybrid
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

import typer
from loguru import logger

from config.settings import settings
from pipeline.embeddings import get_embedding_generator
from pipeline.generation.llama3 import Llama3Client
from pipeline.generation.llm_client import BaseLLMClient, GenerationResponse
from pipeline.generation.mistral import MistralClient
from pipeline.ingestion import load_bm25_corpus
from pipeline.retrieval.dense import DenseRetriever
from pipeline.retrieval.hybrid import HybridRetriever
from pipeline.vectorstore import RetrievedChunk, get_vector_store
from telemetry.logger import TelemetryLogger, build_record, get_telemetry_logger
from telemetry.timer import Timer
from telemetry.token_counter import TokenCounter, TokenUsage

# Evaluation imports are guarded so the pipeline runs without RAGAS
# being initialised (useful for fast integration tests).
try:
    from evaluation.hallucination_score import compute_hallucination_risk
    from evaluation.ragas_runner import RAGASRunner, get_ragas_runner

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


LLMChoice = Literal["llama3", "mistral"]
RetrievalChoice = Literal["dense", "hybrid"]

app = typer.Typer(add_completion=False)


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class RAGResult:
    """
    Complete result of one RAG query execution.

    Attributes
    ----------
    query               : Original query string.
    answer              : Generated response text.
    retrieved_chunks    : Ordered list of retrieved passages.
    retrieval_telemetry : Dict of retrieval timing and score metrics.
    generation_response : Raw GenerationResponse from the LLM client.
    token_usage         : Token counts and estimated cost.
    context_relevance   : RAGAS context relevance score [0, 1].
    answer_faithfulness : RAGAS answer faithfulness score [0, 1].
    answer_correctness  : RAGAS answer correctness score [0, 1] (requires ground truth).
    hallucination_risk  : Composite hallucination risk score [0, 1].
    e2e_ms              : Total end-to-end wall-clock time (ms).
    telemetry_path      : Path to the persisted telemetry JSON file.
    """

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_telemetry: dict
    generation_response: GenerationResponse
    token_usage: TokenUsage
    context_relevance: float | None = None
    answer_faithfulness: float | None = None
    answer_correctness: float | None = None
    hallucination_risk: float | None = None
    e2e_ms: float = 0.0
    telemetry_path: str = ""
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted terminal summary string."""
        lines = [
            "─" * 56,
            f" Query : {self.query[:70]}",
            "─" * 56,
            f" Retrieved chunks : {len(self.retrieved_chunks)} "
            f"(strategy: {self.retrieval_telemetry.get('strategy', '?')})",
        ]
        for i, c in enumerate(self.retrieved_chunks[:3], 1):
            lines.append(f"   [{i}] score={c.score:.3f} | {c.text[:80].strip()}…")
        lines += [
            "",
            f" Answer : {self.answer[:200].strip()}",
            "",
            " ── Telemetry ──────────────────────────────────────",
            f"  Retrieval latency  : {self.retrieval_telemetry.get('retrieval_ms', 0):.1f} ms",
            f"  Generation latency : {self.generation_response.generation_ms:.1f} ms",
            f"  End-to-end latency : {self.e2e_ms:.1f} ms",
            f"  Prompt tokens      : {self.token_usage.prompt_tokens}",
            f"  Completion tokens  : {self.token_usage.completion_tokens}",
            f"  Estimated cost     : ${self.token_usage.estimated_cost_usd:.6f}",
            "",
            " ── RAGAS Evaluation ───────────────────────────────",
            f"  Context Relevance  : {_fmt(self.context_relevance)}",
            f"  Answer Faithfulness: {_fmt(self.answer_faithfulness)}",
            f"  Answer Correctness : {_fmt(self.answer_correctness)}",
            f"  Hallucination Risk : {_risk_label(self.hallucination_risk)}",
            "─" * 56,
            f" Telemetry → {self.telemetry_path}",
            "─" * 56,
        ]
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end RAG query pipeline.

    Parameters
    ----------
    llm_model          : ``"llama3"`` or ``"mistral"``.
    retrieval_strategy : ``"dense"`` or ``"hybrid"``.
    top_k              : Number of chunks to retrieve per query.
    run_evaluation     : Whether to run RAGAS metrics after generation.
    telemetry_logger   : TelemetryLogger instance.
    dataset            : Dataset label for telemetry (e.g. ``"msmarco"``).
    experiment_id      : Optional experiment identifier for batch runs.
    """

    def __init__(
        self,
        llm_model: LLMChoice = "llama3",
        retrieval_strategy: RetrievalChoice = "hybrid",
        top_k: int | None = None,
        run_evaluation: bool = True,
        telemetry_logger: TelemetryLogger | None = None,
        dataset: str = "interactive",
        experiment_id: str | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k or settings.top_k
        self.run_evaluation = run_evaluation and _RAGAS_AVAILABLE
        self.dataset = dataset
        self.experiment_id = experiment_id
        self._tel_logger = telemetry_logger or get_telemetry_logger()
        self._token_counter = TokenCounter()

        # ── LLM client ────────────────────────────────────────────────────────
        self._llm: BaseLLMClient = Llama3Client() if llm_model == "llama3" else MistralClient()
        logger.info(f"LLM client: {self._llm}")

        # ── Retriever ─────────────────────────────────────────────────────────
        store = get_vector_store()
        embedder = get_embedding_generator()

        if retrieval_strategy == "dense":
            self._retriever = DenseRetriever(
                vector_store=store,
                embedding_generator=embedder,
                top_k=self.top_k,
            )
        else:
            hybrid = HybridRetriever(
                vector_store=store,
                embedding_generator=embedder,
                top_k=self.top_k,
            )
            # Load the persisted BM25 corpus and build the index.
            ids, texts = load_bm25_corpus()
            hybrid.build_bm25_index(ids, texts)
            self._retriever = hybrid

        logger.info(
            f"RAG pipeline ready — "
            f"LLM: {llm_model}, Retrieval: {retrieval_strategy}, top_k: {self.top_k}"
        )

        # ── RAGAS runner ──────────────────────────────────────────────────────
        self._ragas: RAGASRunner | None = None
        if self.run_evaluation:
            self._ragas = get_ragas_runner()

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        ground_truth: str | None = None,
        query_id: str | None = None,
        query_type: str = "unknown",
    ) -> RAGResult:
        """
        Execute a full RAG query and return a structured result.

        Parameters
        ----------
        query_text   : The user question.
        ground_truth : Reference answer for RAGAS answer_correctness.
        query_id     : Stable identifier (from benchmark dataset).
        query_type   : Category label (e.g. ``"single_hop"``, ``"bridge"``).

        Returns
        -------
        RAGResult
        """
        with Timer("e2e") as e2e_timer:
            # ── Step 1: Retrieve ───────────────────────────────────────────────
            with Timer("retrieval") as _r:
                chunks, retrieval_tel = self._retriever.retrieve(
                    query=query_text,
                    top_k=self.top_k,
                )
            logger.info(
                f"Retrieved {len(chunks)} chunks in {retrieval_tel['retrieval_ms']:.1f} ms."
            )

            logger.info(f"Query: {query_text}")
            logger.info(f"Ground truth: {ground_truth or 'N/A'}")

            # ── Step 2: Generate ──────────────────────────────────────────────
            context_texts = [c.text for c in chunks]
            gen_response: GenerationResponse = self._llm.generate(
                query=query_text,
                context_chunks=context_texts,
            )
            logger.info(
                f"Generated {gen_response.completion_tokens} tokens "
                f"in {gen_response.generation_ms:.1f} ms."
            )

            # ── Step 3: Token usage ───────────────────────────────────────────
            token_usage = self._token_counter.from_generation_response(gen_response)

            # ── Step 4: Evaluate ──────────────────────────────────────────────
            ctx_rel = faith = correctness = risk = None
            if self._ragas and chunks:
                scores = self._ragas.evaluate(
                    query=query_text,
                    answer=gen_response.text,
                    contexts=[c.text for c in chunks],
                    ground_truth=ground_truth,
                )
                ctx_rel = scores.get("context_relevance")
                faith = scores.get("answer_faithfulness")
                correctness = scores.get("answer_correctness")
                risk = compute_hallucination_risk(
                    faithfulness=faith,
                    context_relevance=ctx_rel,
                )

        e2e_ms = e2e_timer.elapsed_ms

        # ── Step 5: Build + persist telemetry record ──────────────────────────
        record = build_record(
            query_id=query_id,
            query=query_text,
            dataset=self.dataset,
            query_type=query_type,
            llm_model=self.llm_model,
            retrieval_strategy=self.retrieval_strategy,
            retrieved_chunks=chunks,
            retrieval_ms=retrieval_tel["retrieval_ms"],
            embed_query_ms=retrieval_tel.get("embed_query_ms", 0.0),
            answer=gen_response.text,
            generation_ms=gen_response.generation_ms,
            token_usage=token_usage,
            context_relevance=ctx_rel,
            answer_faithfulness=faith,
            answer_correctness=correctness,
            hallucination_risk=risk,
            ground_truth=ground_truth,
            experiment_id=self.experiment_id,
        )
        tel_path = self._tel_logger.log(record)

        return RAGResult(
            query=query_text,
            answer=gen_response.text,
            retrieved_chunks=chunks,
            retrieval_telemetry=retrieval_tel,
            generation_response=gen_response,
            token_usage=token_usage,
            context_relevance=ctx_rel,
            answer_faithfulness=faith,
            answer_correctness=correctness,
            hallucination_risk=risk,
            e2e_ms=e2e_ms,
            telemetry_path=str(tel_path),
            metadata={"query_id": query_id, "query_type": query_type},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "N/A (no ground truth)"


def _risk_label(v: float | None) -> str:
    if v is None:
        return "N/A"
    label = "LOW" if v < 0.35 else "MEDIUM" if v < 0.65 else "HIGH"
    return f"{label}  ({v:.4f})"


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    query: str = typer.Argument(..., help="The question to ask the RAG system."),
    llm: LLMChoice = typer.Option("llama3", help="LLM model: llama3 | mistral"),
    retrieval: RetrievalChoice = typer.Option("hybrid", help="Retrieval strategy: dense | hybrid"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
    ground_truth: str = typer.Option("", help="Optional reference answer for evaluation."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip RAGAS evaluation."),
) -> None:
    """Run a single query through the RAG pipeline and print the result."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    pipeline = RAGPipeline(
        llm_model=llm,
        retrieval_strategy=retrieval,
        top_k=top_k,
        run_evaluation=not no_eval,
    )
    result = pipeline.query(
        query_text=query,
        ground_truth=ground_truth or None,
    )
    print(result.summary())


if __name__ == "__main__":
    app()
