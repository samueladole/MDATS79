"""
RAGScope — Timer
=================
Context-manager-based wall-clock timer for profiling individual pipeline
stages. Each stage's elapsed time is recorded in milliseconds and surfaced
through the telemetry logger.
"""

from __future__ import annotations

import time


class Timer:
    """
    Context manager that records the elapsed wall-clock time of a code block.

    Usage
    -----
        with Timer("retrieval") as t:
            chunks = retriever.retrieve(query)
        print(t.elapsed_ms)   # e.g. 342.1

    Attributes
    ----------
    name        : Label for this timing stage (used in telemetry logs).
    elapsed_ms  : Wall-clock duration in milliseconds (set on __exit__).
    """

    def __init__(self, name: str = "") -> None:
        self.name: str = name
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed_ms={self.elapsed_ms})"
