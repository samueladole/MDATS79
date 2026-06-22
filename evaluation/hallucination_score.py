"""
RAGScope — Hallucination Risk Scorer
======================================
Computes a composite hallucination risk score from RAGAS metrics.

Formula (from the proposal)
---------------------------
    hallucination_risk = 1 - (0.6 × faithfulness + 0.4 × context_relevance)

Rationale
---------
* Answer faithfulness is weighted more heavily (0.6) because a faithful
  answer — one whose claims are grounded in retrieved context — is the
  primary defence against hallucination in RAG systems (Ji et al., 2023).
* Context relevance receives secondary weight (0.4) because poor retrieval
  (irrelevant context) is the upstream cause of most faithfulness failures;
  if the context is irrelevant, faithfulness cannot be meaningfully assessed
  (Barnett et al., 2024).
* The composite is inverted (1 - ...) so that higher score = higher risk,
  matching the intuitive interpretation of a "risk gauge" in the dashboard.

Risk bands
----------
    [0.00, 0.35) : LOW    — answer likely faithful and well-grounded
    [0.35, 0.65) : MEDIUM — some risk; review retrieved context
    [0.65, 1.00] : HIGH   — likely hallucinated; do not trust answer
"""

from __future__ import annotations

# ── Weights ───────────────────────────────────────────────────────────────────
FAITHFULNESS_WEIGHT = 0.6
CONTEXT_RELEVANCE_WEIGHT = 0.4

# ── Risk band thresholds ──────────────────────────────────────────────────────
LOW_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.65


def compute_hallucination_risk(
    faithfulness: float | None,
    context_relevance: float | None,
) -> float | None:
    """
    Compute the composite hallucination risk score.

    If either input metric is None (i.e. RAGAS failed to compute it),
    the function returns None rather than producing a misleading score.

    Parameters
    ----------
    faithfulness      : RAGAS answer_faithfulness score [0, 1].
    context_relevance : RAGAS context_relevance score [0, 1].

    Returns
    -------
    float | None
        Risk score in [0, 1], or None if inputs are unavailable.
    """
    if faithfulness is None or context_relevance is None:
        return None

    score = 1.0 - (
        FAITHFULNESS_WEIGHT * faithfulness + CONTEXT_RELEVANCE_WEIGHT * context_relevance
    )
    # Clamp to [0, 1] to handle floating-point edge cases
    return round(max(0.0, min(1.0, score)), 4)


def risk_band(score: float | None) -> str:
    """
    Map a hallucination risk score to its categorical band label.

    Parameters
    ----------
    score : Output of ``compute_hallucination_risk``.

    Returns
    -------
    str
        ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, or ``"UNKNOWN"`` if score is None.
    """
    if score is None:
        return "UNKNOWN"
    if score < LOW_THRESHOLD:
        return "LOW"
    if score < MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def risk_colour(score: float | None) -> str:
    """
    Map a risk band to a hex colour for dashboard visualisation.

    Returns
    -------
    str  Hex colour code.
    """
    band = risk_band(score)
    return {
        "LOW": "#2ecc71",  # green
        "MEDIUM": "#f39c12",  # amber
        "HIGH": "#e74c3c",  # red
        "UNKNOWN": "#95a5a6",  # grey
    }[band]
