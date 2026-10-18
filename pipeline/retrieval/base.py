"""
Common interface for retrieval strategies.

Dense, hybrid and graph retrievers were previously related only by duck typing:
each happened to expose ``retrieve(query, top_k)``, and the orchestrator selected
between them with an inline ``if/else``. That arrangement had a sharp edge — an
unrecognised strategy name fell through to the ``else`` branch and silently ran a
*different* retriever than the one named in the telemetry record.

This module makes the contract explicit, mirroring ``BaseLLMClient`` in
``pipeline/generation/llm_client.py``: a concrete ``__init__``, one abstract
identity property, and one abstract worker method.

Deliberately NOT in the base class
----------------------------------
* **Telemetry assembly.** Each retriever still builds its own telemetry dict.
  Hoisting the shared keys into a template method would change ``retrieval_ms``
  from "sum of stage timers" to "wall clock", which would silently shift every
  published latency figure. Deferred as a separate, clearly-labelled change.
* **Index loading.** ``HybridRetriever.build_bm25_index`` and a graph
  retriever's graph load share nothing but being slow. They belong to the
  factories in ``pipeline/retrieval/__init__.py``, not to a forced-symmetric
  ``warmup()`` that only some subclasses need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config.settings import settings
from pipeline.vectorstore import RetrievedChunk


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.

    Subclasses must implement ``strategy_name`` and ``retrieve``.
    """

    #: Scale that ``RetrievedChunk.score`` carries for this strategy.
    #: Scores are NOT comparable across strategies: dense emits cosine
    #: similarity (~0.75), hybrid emits a Reciprocal Rank Fusion score (~0.016).
    #: Recorded in telemetry so downstream analysis cannot silently compare them.
    score_kind: str = "cosine"

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.top_k

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Strategy identifier, e.g. 'dense'.

        Must be a single token: the experiment runner's ``--conditions`` parser
        splits condition labels on '_' and expects exactly two parts, so a name
        containing an underscore is silently dropped.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """Retrieve the most relevant chunks for ``query``.

        Returns the ranked chunks plus a telemetry dict which must contain at
        least: ``strategy``, ``top_k``, ``retrieval_ms``, ``chunks_retrieved``,
        ``top_score`` and ``mean_score``.
        """
        ...

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{type(self).__name__}(strategy={self.strategy_name!r}, top_k={self.top_k})"
