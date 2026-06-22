"""
RAGScope — Metrics Aggregation
================================
Aggregates per-query RAGAS scores and telemetry into summary statistics
used by the benchmark analysis and the dashboard comparison view.

Implements the statistical analysis plan from the proposal:
    * Descriptive statistics (mean, std, IQR, min, max)
    * Cohen's d effect sizes for pairwise condition comparisons
    * Pearson correlations between latency and quality metrics

Reference
---------
Cohen, J. (1988) Statistical Power Analysis for the Behavioural Sciences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Descriptive statistics ────────────────────────────────────────────────────


@dataclass
class DescriptiveStats:
    """Summary statistics for a single metric across N queries."""

    metric: str
    n: int
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "q1": round(self.q1, 4),
            "q3": round(self.q3, 4),
            "iqr": round(self.iqr, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
        }


def descriptive_stats(values: list[float], metric: str = "") -> DescriptiveStats:
    """
    Compute descriptive statistics for a list of numeric values.

    Parameters
    ----------
    values : List of metric values (NaN / None are excluded).
    metric : Label for the metric (used in output dicts).
    """
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return DescriptiveStats(metric, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    s = sorted(clean)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    median = _percentile(s, 50)
    q1 = _percentile(s, 25)
    q3 = _percentile(s, 75)

    return DescriptiveStats(
        metric=metric,
        n=n,
        mean=mean,
        std=std,
        median=median,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        minimum=s[0],
        maximum=s[-1],
    )


# ── Effect size: Cohen's d ─────────────────────────────────────────────────────


@dataclass
class CohensD:
    """
    Cohen's d effect size between two groups.

    Interpretation (Cohen, 1988):
        |d| < 0.2   : negligible
        0.2 ≤ |d| < 0.5 : small
        0.5 ≤ |d| < 0.8 : medium
        |d| ≥ 0.8   : large
    """

    group_a: str
    group_b: str
    metric: str
    d: float
    magnitude: str  # "negligible" | "small" | "medium" | "large"
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int

    def to_dict(self) -> dict:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "metric": self.metric,
            "d": round(self.d, 4),
            "magnitude": self.magnitude,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


def cohens_d(
    group_a: list[float],
    group_b: list[float],
    label_a: str = "A",
    label_b: str = "B",
    metric: str = "",
) -> CohensD:
    """
    Compute Cohen's d (pooled standard deviation) for two independent groups.

    Parameters
    ----------
    group_a, group_b : Lists of metric values for each condition.
    label_a, label_b : Condition labels for the output record.
    metric           : Metric name for labelling.
    """
    a = [v for v in group_a if v is not None and not math.isnan(v)]
    b = [v for v in group_b if v is not None and not math.isnan(v)]

    if len(a) < 2 or len(b) < 2:
        return CohensD(
            label_a, label_b, metric, 0.0, "negligible", _mean(a), _mean(b), len(a), len(b)
        )

    mean_a, mean_b = _mean(a), _mean(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))

    d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    magnitude = (
        "negligible"
        if abs(d) < 0.2
        else "small"
        if abs(d) < 0.5
        else "medium"
        if abs(d) < 0.8
        else "large"
    )
    return CohensD(label_a, label_b, metric, round(d, 4), magnitude, mean_a, mean_b, len(a), len(b))


# ── Pearson correlation ───────────────────────────────────────────────────────


def pearson_r(x: list[float], y: list[float]) -> float | None:
    """
    Compute Pearson correlation coefficient between two equal-length lists.
    Returns None if fewer than 3 valid paired values.
    """
    pairs = [
        (xi, yi)
        for xi, yi in zip(x, y)
        if xi is not None and yi is not None and not math.isnan(xi) and not math.isnan(yi)
    ]
    n = len(pairs)
    if n < 3:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = _mean(xs), _mean(ys)

    num = sum((xi - mx) * (yi - my) for xi, yi in pairs)
    den = math.sqrt(sum((xi - mx) ** 2 for xi in xs) * sum((yi - my) ** 2 for yi in ys))
    return round(num / den, 4) if den > 0 else None


# ── Condition comparison ──────────────────────────────────────────────────────

EVAL_METRICS = [
    "context_relevance",
    "answer_faithfulness",
    "answer_correctness",
    "hallucination_risk",
    "e2e_ms",
    "total_tokens",
    "estimated_cost_usd",
]


def compare_conditions(
    records_a: list[dict],
    records_b: list[dict],
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Generate a full pairwise comparison between two sets of telemetry records.

    Computes descriptive stats for each metric in both groups and
    Cohen's d for all evaluation metrics.

    Parameters
    ----------
    records_a, records_b : Lists of telemetry record dicts.
    label_a, label_b     : Condition labels (e.g. ``"llama3_dense"``).

    Returns
    -------
    dict
        ``{metric: {"stats_a": ..., "stats_b": ..., "cohens_d": ...}}``
    """
    result = {}
    for metric in EVAL_METRICS:
        vals_a = [r.get(metric) for r in records_a]
        vals_b = [r.get(metric) for r in records_b]
        result[metric] = {
            "stats_a": descriptive_stats(vals_a, metric).to_dict(),
            "stats_b": descriptive_stats(vals_b, metric).to_dict(),
            "cohens_d": cohens_d(vals_a, vals_b, label_a, label_b, metric).to_dict(),
        }
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    idx = (p / 100) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
