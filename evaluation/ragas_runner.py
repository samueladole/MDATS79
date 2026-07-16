"""
RAGScope — RAGAS Runner
========================
Wraps the RAGAS evaluation framework to compute per-query quality metrics.

Metrics computed
----------------
* context_relevance   : Proportion of retrieved context relevant to the query.
* answer_faithfulness : Degree to which the answer is grounded in the context.
* answer_correctness  : Semantic match with the ground-truth answer (when provided).

The auxiliary LLM used for RAGAS metric computation is set to
``settings.ragas_judge_model`` (default: ``qwen2.5:7b``) and is held constant
across ALL experimental conditions to ensure comparability. This is a key
methodological control identified in the proposal (Es et al., 2023).

RAGAS ≥ 0.2 uses the ``EvaluationDataset`` API. This wrapper abstracts the
version differences so the rest of the codebase remains stable.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from loguru import logger

from config.settings import settings


class _SentenceTransformerEmbeddings(Embeddings):
    """Adapts the project's ``EmbeddingGenerator`` (all-MiniLM-L6-v2) to the
    LangChain ``Embeddings`` interface so it can back RAGAS's answer
    correctness metric without requiring a separate Ollama embedding model."""

    def __init__(self) -> None:
        from pipeline.embeddings import get_embedding_generator

        self._generator = get_embedding_generator()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._generator.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._generator.embed_query(text)


class RAGASRunner:
    """
    Computes RAGAS evaluation metrics for a single query.

    The judge LLM is initialised once and reused across all calls
    to avoid repeated model loading overhead.

    Parameters
    ----------
    judge_model : Ollama model name for the RAGAS judge. Default from settings.
    base_url    : Ollama base URL. Default from settings.
    """

    def __init__(
        self,
        judge_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.judge_model = judge_model or settings.ragas_judge_model
        self.base_url = base_url or settings.ollama_base_url

        logger.info(f"Initialising RAGAS runner (judge model: '{self.judge_model}') …")
        self._llm, self._embeddings = self._build_judge()
        logger.info("RAGAS runner ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float | None]:
        """
        Compute RAGAS metrics for a single query-answer-context triple.

        Parameters
        ----------
        query        : The original user question.
        answer       : The generated response to evaluate.
        contexts     : List of retrieved passage texts used as context.
        ground_truth : Reference answer for ``answer_correctness``.
                       If None, ``answer_correctness`` will be None.

        Returns
        -------
        dict with keys:
            ``context_relevance``, ``answer_faithfulness``,
            ``answer_correctness`` (None if no ground truth supplied).
        """
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            AnswerCorrectness,  # type: ignore
            ContextRelevance,  # type: ignore
            Faithfulness,  # type: ignore
        )

        results: dict[str, float | None] = {
            "context_relevance": None,
            "answer_faithfulness": None,
            "answer_correctness": None,
        }

        try:
            # Build the metrics list conditionally
            metrics = [
                ContextRelevance(llm=self._llm),
                Faithfulness(llm=self._llm),
            ]
            if ground_truth:
                metrics.append(AnswerCorrectness(llm=self._llm, embeddings=self._embeddings))

            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth or "",
            )
            dataset = EvaluationDataset(samples=[sample])
            scores = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=True,
            )

            logger.debug(f"Raw RAGAS scores: {scores}")

            df = scores.to_pandas()
            if not df.empty:
                row = df.iloc[0]
                results["context_relevance"] = _safe_float(row.get("nv_context_relevance"))
                results["answer_faithfulness"] = _safe_float(row.get("faithfulness"))
                if ground_truth:
                    results["answer_correctness"] = _safe_float(row.get("answer_correctness"))

        except Exception as exc:
            logger.warning(f"RAGAS evaluation failed: {exc}. Returning None scores.")

        logger.debug(
            f"RAGAS scores — ctx_rel={results['context_relevance']}, "
            f"faithfulness={results['answer_faithfulness']}, "
            f"correctness={results['answer_correctness']}"
        )
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_judge(self):
        """
        Build the RAGAS-compatible LLM and embeddings objects.

        RAGAS ≥ 0.2 uses LangChain-style wrappers. The judge LLM is routed
        through our local Ollama instance via ``langchain_ollama``. Embeddings
        reuse the project's existing ``all-MiniLM-L6-v2`` SentenceTransformer
        (``pipeline/embeddings.py``) rather than Ollama: ``self.judge_model``
        is a chat model (e.g. ``qwen2.5:7b``) and Ollama's embeddings endpoint
        rejects models that weren't loaded in embedding-serving mode.
        """
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        native_llm = ChatOllama(
            model=self.judge_model,
            base_url=self.base_url,
            temperature=0.0,
            num_predict=settings.ragas_max_tokens,
        )
        llm = LangchainLLMWrapper(native_llm)
        embeddings = LangchainEmbeddingsWrapper(_SentenceTransformerEmbeddings())
        return llm, embeddings


def _safe_float(v) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        f = float(v)
        return round(f, 4) if not (f != f) else None  # NaN check
    except (TypeError, ValueError):
        return None


# ── Module-level singleton ─────────────────────────────────────────────────────
_runner: RAGASRunner | None = None


def get_ragas_runner() -> RAGASRunner:
    global _runner
    if _runner is None:
        _runner = RAGASRunner()
    return _runner
