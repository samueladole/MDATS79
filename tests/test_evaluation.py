"""Tests for the evaluation layer — metrics, hallucination scoring."""

from __future__ import annotations

import math

from evaluation.hallucination_score import (
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    compute_hallucination_risk,
    risk_band,
    risk_colour,
)
from evaluation.metrics import cohens_d, descriptive_stats, pearson_r


class TestDescriptiveStats:
    def test_basic(self):
        stats = descriptive_stats([0.1, 0.2, 0.3, 0.4, 0.5])
        assert stats.n == 5
        assert math.isclose(stats.mean, 0.3, abs_tol=1e-6)
        assert stats.minimum == 0.1
        assert stats.maximum == 0.5

    def test_empty_returns_zeros(self):
        stats = descriptive_stats([])
        assert stats.n == 0
        assert stats.mean == 0.0

    def test_none_values_excluded(self):
        stats = descriptive_stats([0.5, None, 0.7, None])
        assert stats.n == 2
        assert math.isclose(stats.mean, 0.6, abs_tol=1e-6)

    def test_single_value(self):
        stats = descriptive_stats([0.75])
        assert stats.n == 1
        assert stats.std == 0.0

    def test_to_dict_keys(self):
        d = descriptive_stats([0.1, 0.5, 0.9]).to_dict()
        expected = {"metric", "n", "mean", "std", "median", "q1", "q3", "iqr", "min", "max"}
        assert set(d.keys()) == expected


class TestCohensD:
    def test_identical_groups(self):
        a = [0.5, 0.6, 0.7, 0.8]
        b = [0.5, 0.6, 0.7, 0.8]
        result = cohens_d(a, b)
        assert result.d == 0.0
        assert result.magnitude == "negligible"

    def test_large_effect(self):
        a = [0.9, 0.9, 0.9, 0.9, 0.9]
        b = [0.1, 0.1, 0.1, 0.1, 0.1]
        result = cohens_d(a, b)
        assert result.magnitude == "large"
        assert result.d > 0

    def test_insufficient_data(self):
        result = cohens_d([0.5], [0.6])
        assert result.d == 0.0

    def test_to_dict(self):
        d = cohens_d([0.5, 0.6], [0.7, 0.8]).to_dict()
        assert "d" in d and "magnitude" in d


class TestPearsonR:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_r(x, y)
        assert math.isclose(r, 1.0, abs_tol=1e-6)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        r = pearson_r(x, y)
        assert math.isclose(r, -1.0, abs_tol=1e-6)

    def test_insufficient_data(self):
        assert pearson_r([1, 2], [3, 4]) is None

    def test_none_values_excluded(self):
        x = [1, None, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_r(x, y)
        assert r is not None


class TestHallucinationScore:
    def test_zero_risk_perfect_scores(self):
        risk = compute_hallucination_risk(faithfulness=1.0, context_relevance=1.0)
        assert risk == 0.0

    def test_max_risk_zero_scores(self):
        risk = compute_hallucination_risk(faithfulness=0.0, context_relevance=0.0)
        assert risk == 1.0

    def test_none_input_returns_none(self):
        assert compute_hallucination_risk(None, 0.8) is None
        assert compute_hallucination_risk(0.8, None) is None
        assert compute_hallucination_risk(None, None) is None

    def test_formula_correctness(self):
        # 1 - (0.6 * 0.8 + 0.4 * 0.6) = 1 - (0.48 + 0.24) = 0.28
        risk = compute_hallucination_risk(faithfulness=0.8, context_relevance=0.6)
        assert math.isclose(risk, 0.28, abs_tol=1e-4)

    def test_clamped_to_zero_one(self):
        # Floating point edge case
        risk = compute_hallucination_risk(faithfulness=1.0, context_relevance=1.0)
        assert 0.0 <= risk <= 1.0

    def test_risk_band_low(self):
        assert risk_band(0.1) == "LOW"
        assert risk_band(LOW_THRESHOLD - 0.01) == "LOW"

    def test_risk_band_medium(self):
        assert risk_band(LOW_THRESHOLD) == "MEDIUM"
        assert risk_band(0.5) == "MEDIUM"

    def test_risk_band_high(self):
        assert risk_band(MEDIUM_THRESHOLD) == "HIGH"
        assert risk_band(0.9) == "HIGH"

    def test_risk_band_none(self):
        assert risk_band(None) == "UNKNOWN"

    def test_risk_colour_returns_hex(self):
        for score in [0.1, 0.5, 0.9, None]:
            colour = risk_colour(score)
            assert colour.startswith("#")
            assert len(colour) == 7
