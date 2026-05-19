"""Unit tests for common.metrics — AUT formulation correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.metrics import aut, aut_with_bootstrap, basic_metrics, tpr_at_fpr  # noqa: E402


class TestAUT:
    def test_constant_curve_equals_value(self):
        # AUT of a constant series equals the constant.
        assert aut([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
        assert aut([1.0, 1.0]) == pytest.approx(1.0)

    def test_linear_decay_midpoint(self):
        # AUT of [1.0, 0.0] (linear decay over 2 points) equals 0.5.
        assert aut([1.0, 0.0]) == pytest.approx(0.5)

    def test_three_point_trapezoid(self):
        # AUT of [1.0, 0.5, 0.0] = average of (1+0.5)/2 and (0.5+0)/2 = 0.5.
        assert aut([1.0, 0.5, 0.0]) == pytest.approx(0.5)

    def test_rejects_single_point(self):
        with pytest.raises(ValueError):
            aut([1.0])

    def test_rejects_two_d(self):
        with pytest.raises(ValueError):
            aut([[1.0, 2.0], [3.0, 4.0]])

    def test_realistic_paper_series(self):
        # Manuscript Table 5 EMBER 2017 -> BODMAS series, LGBM.
        series = [95.59, 98.71, 98.68, 98.70, 99.05, 98.26, 97.68,
                  98.31, 98.52, 99.08, 98.53, 99.14, 99.55, 99.23]
        result = aut(series)
        # Should be close to mean since values are tightly clustered.
        assert result == pytest.approx(sum(series) / len(series), abs=0.5)
        assert 95.0 < result < 99.5


class TestAUTBootstrap:
    def test_zero_variance_when_all_seeds_equal(self):
        curves = np.full((10, 13), 0.97)
        stats = aut_with_bootstrap(curves, n_bootstrap=200)
        assert stats["mean"] == pytest.approx(0.97)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["ci_low"] == pytest.approx(0.97)
        assert stats["ci_high"] == pytest.approx(0.97)

    def test_ci_brackets_mean_for_noisy_seeds(self):
        rng = np.random.default_rng(42)
        truth = 0.97
        curves = truth + rng.normal(0, 0.005, size=(10, 13))
        stats = aut_with_bootstrap(curves, n_bootstrap=2000, rng=np.random.default_rng(0))
        assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
        # CI width should be < 1 pp for a stable signal with 10 seeds.
        assert (stats["ci_high"] - stats["ci_low"]) < 0.01


class TestBasicMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = basic_metrics(y_true, y_pred)
        assert m["f1"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["fnr"] == pytest.approx(0.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = basic_metrics(y_true, y_pred)
        assert m["f1"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(1.0)
        assert m["fnr"] == pytest.approx(1.0)


class TestTPRatFPR:
    def test_perfectly_separable(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        # Even at 1% FPR, all malware are above the threshold.
        assert tpr_at_fpr(y_true, y_proba, target_fpr=0.01) == pytest.approx(1.0)
