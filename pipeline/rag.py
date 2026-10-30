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
from pipeline.generation import BaseLLMClient, GenerationResponse, build_llm_client
from pipeline.knowledge_base import get_knowledge_base
from pipeline.retrieval import BaseRetriever, build_retriever
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


RetrievalChoice = Literal["dense", "hybrid", "graph"]

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
    evaluation_ms       : Wall-clock time spent in RAGAS evaluation (ms); 0 if skipped.
    e2e_ms              : Total end-to-end wall-clock time (ms) — includes evaluation.
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
    evaluation_ms: float = 0.0
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
            f"  Evaluation latency : {self.evaluation_ms:.1f} ms",
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


@dataclass
class DraftAnswer:
    """An answer that has been generated but not yet evaluated or logged.

    Returned by ``RAGPipeline.answer`` and consumed by ``RAGPipeline.evaluate``.
    ``elapsed_ms`` is the wall-clock time of retrieval plus generation.
    """

    query: str
    chunks: list[RetrievedChunk]
    retrieval_telemetry: dict
    generation_response: GenerationResponse
    token_usage: TokenUsage
    elapsed_ms: float

    @property
    def answer(self) -> str:
        return self.generation_response.text


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end RAG query pipeline.

    Parameters
    ----------
    llm_model          : Ollama model tag. Defaults to ``settings.default_llm``.
    retrieval_strategy : ``"dense"`` or ``"hybrid"``.
    top_k              : Number of chunks to retrieve per query.
    run_evaluation     : Whether to run RAGAS metrics after generation.
    telemetry_logger   : TelemetryLogger instance.
    dataset            : Dataset label for telemetry (e.g. ``"bioasq_phase_a"``).
    experiment_id      : Optional experiment identifier for batch runs.
    knowledge_base     : ``"bioasq"`` (the benchmark corpus) or ``"documents"``
                         (your own ingested files).
    """

    def __init__(
        self,
        llm_model: str | None = None,
        retrieval_strategy: RetrievalChoice = "hybrid",
        top_k: int | None = None,
        run_evaluation: bool = True,
        telemetry_logger: TelemetryLogger | None = None,
        dataset: str = "interactive",
        experiment_id: str | None = None,
        knowledge_base: str = "bioasq",
    ) -> None:
        llm_model = llm_model or settings.default_llm
        self.llm_model = llm_model
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k or settings.top_k
        self.run_evaluation = run_evaluation and _RAGAS_AVAILABLE
        self.dataset = dataset
        self.experiment_id = experiment_id
        self._tel_logger = telemetry_logger or get_telemetry_logger()
        self._token_counter = TokenCounter()

        # ── LLM client ────────────────────────────────────────────────────────
        self._llm: BaseLLMClient = build_llm_client(llm_model)
        logger.info(f"LLM client: {self._llm}")

        # ── Retriever ─────────────────────────────────────────────────────────
        kb = get_knowledge_base(knowledge_base)
        if retrieval_strategy == "graph" and not kb.supports_graph:
            raise ValueError(
                f"Graph retrieval needs a knowledge graph, and there is none for "
                f"'{kb.label}'. Use dense or hybrid retrieval."
            )
        self.knowledge_base = kb.name
        store = get_vector_store(kb.collection)
        embedder = get_embedding_generator()

        # build_retriever raises on an unknown strategy. The previous inline
        # if/else fell through to hybrid for any name that was not "dense",
        # which produced telemetry labelled with a strategy that never ran.
        self._retriever: BaseRetriever = build_retriever(
            retrieval_strategy,
            store=store,
            embedder=embedder,
            top_k=self.top_k,
            bm25_corpus_path=kb.bm25_path,
        )

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

        Equivalent to ``evaluate(answer(query_text), ...)``. Callers that want
        to show the answer before the slow evaluation stage finishes call the
        two halves themselves.

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
        draft = self.answer(query_text, ground_truth=ground_truth)
        return self.evaluate(
            draft, ground_truth=ground_truth, query_id=query_id, query_type=query_type
        )

    def answer(self, query_text: str, ground_truth: str | None = None) -> DraftAnswer:
        """Retrieve context and generate an answer, without evaluating it."""
        with Timer("answer") as answer_timer:
            # ── Step 1: Retrieve ───────────────────────────────────────────────
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

        return DraftAnswer(
            query=query_text,
            chunks=chunks,
            retrieval_telemetry=retrieval_tel,
            generation_response=gen_response,
            token_usage=token_usage,
            elapsed_ms=answer_timer.elapsed_ms,
        )

    def evaluate(
        self,
        draft: DraftAnswer,
        ground_truth: str | None = None,
        query_id: str | None = None,
        query_type: str = "unknown",
    ) -> RAGResult:
        """Score a draft answer, persist its telemetry record and return the result."""
        chunks = draft.chunks
        gen_response = draft.generation_response

        # ── Step 4: Evaluate ──────────────────────────────────────────────────
        ctx_rel = faith = correctness = risk = None
        with Timer("evaluation") as eval_timer:
            if self._ragas and chunks:
                scores = self._ragas.evaluate(
                    query=draft.query,
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
        evaluation_ms = eval_timer.elapsed_ms if self._ragas else 0.0

        # e2e_ms covers retrieval + generation + evaluation. It is the sum of
        # the two halves' own timers, not the wall clock between them, so time
        # a caller spends between answer() and evaluate() (the dashboard
        # rendering the answer) is not counted as pipeline latency. RAGAS
        # evaluation can dominate total latency, and leaving it out would
        # silently understate e2e_ms.
        e2e_ms = round(draft.elapsed_ms + eval_timer.elapsed_ms, 2)

        # ── Step 5: Build + persist telemetry record ──────────────────────────
        record = build_record(
            query_id=query_id,
            query=draft.query,
            dataset=self.dataset,
            query_type=query_type,
            llm_model=self.llm_model,
            retrieval_strategy=self.retrieval_strategy,
            retrieved_chunks=chunks,
            retrieval_telemetry=draft.retrieval_telemetry,
            answer=gen_response.text,
            generation_ms=gen_response.generation_ms,
            evaluation_ms=evaluation_ms,
            e2e_ms=e2e_ms,
            token_usage=draft.token_usage,
            context_relevance=ctx_rel,
            answer_faithfulness=faith,
            answer_correctness=correctness,
            hallucination_risk=risk,
            ground_truth=ground_truth,
            experiment_id=self.experiment_id,
        )
        tel_path = self._tel_logger.log(record)

        return RAGResult(
            query=draft.query,
            answer=gen_response.text,
            retrieved_chunks=chunks,
            retrieval_telemetry=draft.retrieval_telemetry,
            generation_response=gen_response,
            token_usage=draft.token_usage,
            context_relevance=ctx_rel,
            answer_faithfulness=faith,
            answer_correctness=correctness,
            hallucination_risk=risk,
            evaluation_ms=evaluation_ms,
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
    llm: str = typer.Option("", help="Ollama model tag, e.g. llama3, mistral, llama3.2:3b. Defaults to DEFAULT_LLM."),
    retrieval: RetrievalChoice = typer.Option("hybrid", help="Retrieval strategy: dense | hybrid"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
    ground_truth: str = typer.Option("", help="Optional reference answer for evaluation."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip RAGAS evaluation."),
) -> None:
    """Run a single query through the RAG pipeline and print the result."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    pipeline = RAGPipeline(
        llm_model=llm or None,
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
