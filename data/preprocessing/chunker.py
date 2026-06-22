"""
RAGScope — Document Chunker
============================
Splits cleaned documents into fixed-size, overlapping chunks suitable
for dense embedding and vector retrieval.

Design choices
--------------
* Chunks are split on token boundaries (via tiktoken) rather than
  character counts, ensuring the embedding model never receives input
  that exceeds its context window.
* A configurable overlap window preserves cross-boundary context,
  which is particularly important for multi-hop HotpotQA passages
  that often carry reasoning chains across sentence boundaries.
* Each chunk is returned as a ``Chunk`` dataclass that carries its
  source document identifier and positional metadata — required by
  the telemetry layer to log which specific chunks were retrieved
  for each query.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import tiktoken

from config.settings import settings

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Chunk:
    """
    A single text chunk ready for embedding and storage.

    Attributes
    ----------
    text        : The chunk text content.
    doc_id      : Unique identifier of the source document.
    chunk_index : Zero-based position of this chunk within the document.
    start_token : Token offset of the first token in the source document.
    end_token   : Token offset (exclusive) of the last token.
    metadata    : Arbitrary key-value pairs propagated from the source
                  document (e.g. dataset name, passage title).
    """

    text: str
    doc_id: str
    chunk_index: int
    start_token: int
    end_token: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Stable unique identifier for this chunk: ``{doc_id}::chunk_{index}``."""
        return f"{self.doc_id}::chunk_{self.chunk_index}"


# ── Chunker ───────────────────────────────────────────────────────────────────


class TokenChunker:
    """
    Splits document text into overlapping token-bounded chunks.

    Uses the ``cl100k_base`` tokeniser (same vocabulary as
    text-embedding-ada-002 and most modern open-source embedding models)
    to ensure consistent token counting across pipeline stages.

    Parameters
    ----------
    chunk_size    : Maximum tokens per chunk. Default from settings.
    chunk_overlap : Token overlap between consecutive chunks. Default from settings.
    """

    # cl100k_base is compatible with all-MiniLM and most sentence-transformers.
    _ENCODING_NAME = "cl100k_base"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )

        self._enc = tiktoken.get_encoding(self._ENCODING_NAME)

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Chunk a single document into a list of ``Chunk`` objects.

        Parameters
        ----------
        text     : Cleaned document text.
        doc_id   : Unique source document identifier.
        metadata : Optional dict propagated to each chunk (e.g. ``{"dataset": "msmarco"}``).

        Returns
        -------
        list[Chunk]
            Ordered list of chunks. Empty list if ``text`` is empty.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        tokens: list[int] = self._enc.encode(text)
        chunks: list[Chunk] = []

        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    start_token=start,
                    end_token=end,
                    metadata=metadata,
                )
            )

            if end == len(tokens):
                break

            start += step
            chunk_index += 1

        return chunks

    def chunk_many(
        self,
        documents: list[dict],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> Iterator[Chunk]:
        """
        Chunk a list of document dicts, yielding chunks one at a time.

        Parameters
        ----------
        documents       : List of dicts with at minimum ``id_field`` and ``text_field``.
        id_field        : Dict key containing the document identifier.
        text_field      : Dict key containing the document text.
        metadata_fields : Additional dict keys to carry forward as chunk metadata.

        Yields
        ------
        Chunk
        """
        metadata_fields = metadata_fields or []
        for doc in documents:
            doc_id = str(doc[id_field])
            text = doc[text_field]
            metadata = {k: doc.get(k) for k in metadata_fields if k in doc}

            yield from self.chunk(text, doc_id, metadata)
