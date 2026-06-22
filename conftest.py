"""
RAGScope — pytest configuration and shared fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Markers ───────────────────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live services "
        "(ChromaDB, Ollama). Excluded from default run.",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests with significant runtime (benchmark runs).",
    )


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def sample_passages() -> list[dict]:
    """A small list of passage dicts for unit tests."""
    return [
        {
            "id": f"doc_{i}",
            "text": f"This is sample passage number {i}. " * 10,
            "title": f"Title {i}",
            "dataset": "test",
        }
        for i in range(5)
    ]


@pytest.fixture()
def sample_chunks(sample_passages):
    """Pre-chunked version of sample_passages."""
    from data.preprocessing.chunker import TokenChunker

    chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
    chunks = []
    for p in sample_passages:
        chunks.extend(chunker.chunk(p["text"], doc_id=p["id"], metadata={"dataset": p["dataset"]}))
    return chunks


@pytest.fixture()
def tmp_telemetry_dir(tmp_path: Path) -> Path:
    """A temporary directory for telemetry JSON files."""
    d = tmp_path / "telemetry"
    d.mkdir()
    return d
