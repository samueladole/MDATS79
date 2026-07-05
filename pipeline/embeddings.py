"""
RAGScope — Embedding Generator
================================
Wraps SentenceTransformers to produce dense vector representations
for document chunks and queries.

Model used: ``all-MiniLM-L6-v2``
    * 384-dimensional embeddings
    * Max sequence length: 256 tokens
    * Lightweight and fast on CPU — appropriate for local research deployment
    * Compatible with ChromaDB's default distance metric (cosine)

The generator is implemented as a singleton-style class so the model
is loaded once per process and reused across all ingestion and query calls.
"""

from __future__ import annotations

import torch
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingGenerator:
    """
    Generates dense vector embeddings using a SentenceTransformer model.

    Parameters
    ----------
    model_name : SentenceTransformer model name or path.
                 Defaults to ``settings.embedding_model``.
    device     : Inference device — ``"cpu"``, ``"cuda"``, or ``"mps"``.
                 Defaults to ``settings.embedding_device``.
    batch_size : Number of texts encoded per forward pass. Default 64.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}' …")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Parameters
        ----------
        text : Input text (query or passage).

        Returns
        -------
        list[float]
            Embedding vector of length ``self.embedding_dim``.
        """
        vector: np.ndarray = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalised → cosine == dot product
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in batches.

        Parameters
        ----------
        texts : List of input strings.

        Returns
        -------
        list[list[float]]
            List of embedding vectors in the same order as ``texts``.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} texts in batches of {self.batch_size} …")
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 500,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string.

        Semantically identical to ``embed``, but named separately so callers
        can clearly distinguish query embeddings from passage embeddings
        (some models use asymmetric encoding).
        """
        return self.embed(query)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Lazy-initialised on first import to avoid loading the model at test-collection time.
_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Return the module-level singleton, initialising it on first call."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator(device=torch.device("mps") if torch.backends.mps.is_available() else None)
    return _generator
