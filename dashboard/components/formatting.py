"""
Number formatting shared by the dashboard pages and components.
"""

from __future__ import annotations


def format_duration(ms: float | None) -> str:
    """A duration in the unit that reads best.

    Seconds from one second up, milliseconds below:
    ``48011.1 -> "48.01 s"``, ``217.3 -> "217 ms"``, ``8.25 -> "8.3 ms"``,
    ``0.08 -> "0.08 ms"``. Telemetry is recorded in milliseconds; this only
    changes how it is shown.
    """
    if ms is None:
        return "—"
    ms = float(ms)
    if ms >= 1000:
        return f"{ms / 1000:,.2f} s"
    if ms >= 10:
        return f"{ms:,.0f} ms"
    return f"{ms:.2f} ms" if ms < 1 else f"{ms:.1f} ms"
