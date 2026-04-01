"""
test_wcp.py
===========

Tests for WeightedConformalPredictor (WCP).

Coverage targets:
- Basic coverage on synthetic AR(1) data
- beta=1.0 matches standard split conformal
- Exponential weights decrease with age (oldest scores get lowest weight)
- Works with insurance-specific scores (PoissonPearsonScore)
- Window size truncation
- API correctness (fit/calibrate/predict_interval, shapes, error on unfit)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.methods import (
    WeightedConformalPredictor,
    _weighted_conformal_quantile,
    _conformal_quantile,
)
from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    PoissonPearsonScore,
)
from tests.conftest import ConstantForecaster, PoissonMeanForecaster


ALPHA = 0.1
COVERAGE_TOLERANCE = 0.10  # 10pp below nominal; WCP is a soft-weighting method


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_ar1(n: int, phi: float = 0.5, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """AR(1) series: y_t = phi * y_{t-1} + eps_t, eps ~ N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, sigma)
    return y


def make_poisson_series(n: int = 500, lam: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=n).astype(float)


# ---------------------------------------------------------------------------
# _weighted_conformal_quantile utility
# ---------------------------------------------------------------------------

class TestWeightedConformalQuantile:
    def test_uniform_weights_matches_empirical_quantile(self):
        """With uniform weights, weighted quantile == standard empirical quantile."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        n = len(scores)
        weights = np.ones(n) / n
        alpha = 0.1
        q_w = _weighted_conformal_quantile(scores, weights, alpha)
        # Should return value such that 90% of mass <= q_w
        cumsum = np.cumsum(weights[np.argsort(scores)])
        idx = np.searchsorted(cumsum, 1 - alpha, side="left")
        expected = float(np.sort(scores)[idx])
        assert q_w == pytest.approx(expected)

    def test_empty_scores_returns_inf(self):
        q = _weighted_conformal_quantile(np.array([]), np.array([]), alpha=0.1)
        assert q == np.inf

    def test_heavier_weight_on_large_score_raises_quantile(self):
        """If all mass is on the largest score, quantile should equal that score."""
        scores = np.array([1.0, 2.0, 10.0])
        # Put almost all weight on the largest score
        weights = np.array([0.01, 0.01, 0.98])
        q = _weighted_conformal_quantile(scores, weights, alpha=0.5)
        # 0.5 threshold: cumulative up to 1.0 is 0.01, up to 2.0 is 0.02,
        # up to 10.0 is 1.0; so first value where cum >= 0.5 is 10.0
        assert q == pytest.approx(10.0)

    def test_returns_finite_for_reasonable_inputs(self):
        rng = np.random.default_rng(7)
        scores = np.abs(rng.normal(0, 5, 100))
        weights = np.ones(100) / 100
        q = _weighted_conformal_quantile(scores, weights, alpha=0.1)
        assert np.isfinite(q)


# ---------------------------------------------------------------------------
# WeightedConformalPredictor._compute_weights
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_weights_sum_to_one(self):
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.9)
        w = wcp._compute_weights(20)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_most_recent_has_largest_weight(self):
        """Last score (most recent) should always have the highest weight."""
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.9)
        w = wcp._compute_weights(10)
        # w[-1] corresponds to the most recent (exponent = 0, weight = 1)
        assert w[-1] == max(w)

    def test_weights_monotone_increasing(self):
        """Weights should increase from oldest to most recent."""
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.8)
        w = wcp._compute_weights(15)
        assert np.all(np.diff(w) >= 0)

    def test_beta_one_gives_uniform_weights(self):
        """beta=1 should produce exactly uniform weights."""
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=1.0)
        w = wcp._compute_weights(10)
        expected = np.ones(10) / 10
        np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_small_beta_strong_downweighting(self):
        """With small beta, the oldest weight should be much smaller than the newest."""
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.5)
        w = wcp._compute_weights(20)
        # ratio of oldest to newest raw weight = beta^(n-1)
        assert w[0] < w[-1] * 0.01

    def test_single_score_weight_is_one(self):
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.9)
        w = wcp._compute_weights(1)
        assert w[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestWCPAPI:
    def test_raises_before_fit(self):
        wcp = WeightedConformalPredictor(ConstantForecaster())
        y = make_poisson_series(50)
        with pytest.raises(RuntimeError, match="fit"):
            wcp.predict_interval(y)

    def test_calibrate_raises_before_fit(self):
        wcp = WeightedConformalPredictor(ConstantForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            wcp.calibrate(make_poisson_series(20))

    def test_fit_returns_self(self):
        y = make_poisson_series(200)
        wcp = WeightedConformalPredictor(ConstantForecaster())
        result = wcp.fit(y[:100])
        assert result is wcp

    def test_calibrate_returns_self(self):
        y = make_poisson_series(300)
        wcp = WeightedConformalPredictor(ConstantForecaster())
        wcp.fit(y[:100])
        result = wcp.calibrate(y[100:150])
        assert result is wcp

    def test_predict_interval_shape(self):
        y = make_poisson_series(300)
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.95)
        wcp.fit(y[:100])
        wcp.calibrate(y[100:150])
        lower, upper = wcp.predict_interval(y[150:])
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_poisson_series(300)
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.95)
        wcp.fit(y[:100])
        wcp.calibrate(y[100:150])
        lower, upper = wcp.predict_interval(y[150:])
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_no_calibration_data_returns_inf(self):
        """Without any calibration data the first interval must be infinite."""
        y = make_poisson_series(100)
        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.9)
        wcp.fit(y[:50])
        wcp._calibration_scores = []
        lower, upper = wcp.predict_interval(y[50:51])
        assert not np.isfinite(upper[0])

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            WeightedConformalPredictor(ConstantForecaster(), beta=0.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            WeightedConformalPredictor(ConstantForecaster(), alpha=1.1)

    def test_alpha_override_in_predict(self):
        """alpha kwarg in predict_interval should override self.alpha."""
        y = make_poisson_series(400)
        wcp = WeightedConformalPredictor(ConstantForecaster(), alpha=0.5)
        wcp.fit(y[:200])
        wcp.calibrate(y[200:300])
        # alpha=0.5 gives narrower intervals than alpha=0.1
        l_narrow, u_narrow = wcp.predict_interval(y[300:], alpha=0.5)
        l_wide, u_wide = wcp.predict_interval(y[300:], alpha=0.1)
        w_narrow = float(np.mean(u_narrow - l_narrow))
        w_wide = float(np.mean(u_wide - l_wide))
        assert w_narrow <= w_wide


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

class TestWCPCoverage:
    def test_coverage_on_ar1(self):
        """WCP should achieve ~90% coverage on AR(1) data."""
        n_train = 200
        n_cal = 100
        n_test = 400
        y = make_ar1(n_train + n_cal + n_test, phi=0.7, sigma=1.5, seed=11)

        # AR(1) data is centred around zero and takes negative values, so
        # clip_lower must be False — otherwise the lower bound clips at 0
        # and coverage collapses to ~50%.
        score = AbsoluteResidualScore(clip_lower=False)
        wcp = WeightedConformalPredictor(
            ConstantForecaster(), score=score, beta=0.95, alpha=ALPHA
        )
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])
        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"WCP AR(1) coverage {coverage:.3f} < {1 - ALPHA - COVERAGE_TOLERANCE:.3f}"
        )

    def test_coverage_on_poisson(self):
        """WCP should achieve ~90% coverage on stationary Poisson data."""
        n_train = 200
        n_cal = 100
        n_test = 400
        y = make_poisson_series(n_train + n_cal + n_test, lam=10.0, seed=22)

        wcp = WeightedConformalPredictor(
            ConstantForecaster(), beta=0.95, alpha=ALPHA
        )
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])
        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"WCP Poisson coverage {coverage:.3f} < {1 - ALPHA - COVERAGE_TOLERANCE:.3f}"
        )


# ---------------------------------------------------------------------------
# beta=1.0 consistency with standard split conformal
# ---------------------------------------------------------------------------

class TestBetaOneConsistency:
    def test_beta_one_quantile_close_to_standard_cp(self):
        """With beta=1 the WCP quantile should match the standard conformal quantile.

        The two differ in the finite-sample correction:
        - Standard CP uses ceil((1-alpha)(n+1))/n
        - WCP with beta=1 uses uniform weights and the weighted quantile formula

        For large n they converge. We test that intervals are comparable
        (within one score unit on average) rather than bit-for-bit identical.
        """
        n_train = 200
        n_cal = 200
        n_test = 100
        y = make_poisson_series(n_train + n_cal + n_test, lam=12.0, seed=33)

        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=1.0, alpha=ALPHA)
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        l_wcp, u_wcp = wcp.predict_interval(y[n_train + n_cal :])
        w_wcp = float(np.mean(u_wcp - l_wcp))

        # Standard conformal via ACI with gamma=0 (no adaptation)
        from insurance_conformal_ts.methods import ACI
        aci = ACI(ConstantForecaster(), gamma=0.0, window_size=None)
        aci.fit(y[:n_train])
        # Seed ACI calibration manually to match WCP's calibration set
        aci._calibration_scores = list(wcp._calibration_scores[:n_cal])
        l_aci, u_aci = aci.predict_interval(y[n_train + n_cal :], alpha=ALPHA)
        w_aci = float(np.mean(u_aci - l_aci))

        # Widths should be in the same ballpark (within 2 units)
        assert abs(w_wcp - w_aci) < 2.0, (
            f"WCP (beta=1) width {w_wcp:.2f} vs ACI (gamma=0) width {w_aci:.2f} differ by > 2"
        )

    def test_beta_one_achieves_coverage(self):
        """beta=1 should achieve nominal coverage (it is standard split CP)."""
        n_train = 200
        n_cal = 150
        n_test = 500
        y = make_poisson_series(n_train + n_cal + n_test, lam=10.0, seed=44)

        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=1.0, alpha=ALPHA)
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])
        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE


# ---------------------------------------------------------------------------
# Downweighting direction: smaller beta -> smaller intervals on stationary data?
# We check the weaker property: intervals should be valid (correct coverage)
# regardless of beta value, and smaller beta means older scores matter less.
# ---------------------------------------------------------------------------

class TestBetaEffect:
    def test_lower_beta_downweights_older_scores(self):
        """With small beta, WCP ignores old scores. Weights should be concentrated
        on recent observations.
        """
        n = 50
        wcp_aggressive = WeightedConformalPredictor(ConstantForecaster(), beta=0.5)
        wcp_mild = WeightedConformalPredictor(ConstantForecaster(), beta=0.99)

        w_aggressive = wcp_aggressive._compute_weights(n)
        w_mild = wcp_mild._compute_weights(n)

        # Aggressive downweighting: last 10% of scores should hold > 90% of mass
        last_k = n // 10
        assert w_aggressive[-last_k:].sum() > 0.9

        # Mild downweighting: last 10% should hold much less than 90% of mass
        assert w_mild[-last_k:].sum() < 0.5

    def test_narrower_beta_gives_valid_coverage(self):
        """Even with very aggressive downweighting, coverage should remain valid."""
        n_train = 150
        n_cal = 100
        n_test = 300
        y = make_poisson_series(n_train + n_cal + n_test, lam=8.0, seed=55)

        wcp = WeightedConformalPredictor(ConstantForecaster(), beta=0.7, alpha=ALPHA)
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])
        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"WCP beta=0.7 coverage {coverage:.3f} < {1 - ALPHA - COVERAGE_TOLERANCE:.3f}"
        )


# ---------------------------------------------------------------------------
# Insurance-specific scores
# ---------------------------------------------------------------------------

class TestWCPWithInsuranceScores:
    def test_with_poisson_pearson_score(self):
        """WCP should work correctly with PoissonPearsonScore."""
        n_train = 200
        n_cal = 100
        n_test = 300
        y = make_poisson_series(n_train + n_cal + n_test, lam=15.0, seed=66)

        wcp = WeightedConformalPredictor(
            PoissonMeanForecaster(),
            score=PoissonPearsonScore(),
            beta=0.95,
            alpha=ALPHA,
        )
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])

        assert lower.shape == (n_test,)
        assert upper.shape == (n_test,)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"WCP+Poisson coverage {coverage:.3f} below tolerance"
        )


# ---------------------------------------------------------------------------
# Window size
# ---------------------------------------------------------------------------

class TestWCPWindowSize:
    def test_window_size_limits_calibration_set(self):
        """After many updates, calibration set should not exceed window_size."""
        y = make_poisson_series(500)
        window = 30
        wcp = WeightedConformalPredictor(
            ConstantForecaster(), beta=0.9, window_size=window
        )
        wcp.fit(y[:100])
        wcp.predict_interval(y[100:400])
        assert len(wcp._calibration_scores) <= window

    def test_window_size_none_keeps_all_scores(self):
        """With window_size=None, the calibration set grows without bound."""
        n_test = 100
        y = make_poisson_series(300)
        wcp = WeightedConformalPredictor(
            ConstantForecaster(), beta=0.9, window_size=None
        )
        wcp.fit(y[:100])
        initial_cal_len = len(wcp._calibration_scores)
        wcp.predict_interval(y[100 : 100 + n_test])
        assert len(wcp._calibration_scores) == initial_cal_len + n_test

    def test_window_size_coverage_valid(self):
        """WCP with a sliding window should still achieve valid coverage."""
        n_train = 150
        n_cal = 50
        n_test = 300
        y = make_poisson_series(n_train + n_cal + n_test, lam=10.0, seed=77)

        wcp = WeightedConformalPredictor(
            ConstantForecaster(), beta=0.9, alpha=ALPHA, window_size=50
        )
        wcp.fit(y[:n_train])
        wcp.calibrate(y[n_train : n_train + n_cal])
        lower, upper = wcp.predict_interval(y[n_train + n_cal :])
        coverage = float(
            np.mean(
                (y[n_train + n_cal :] >= lower) & (y[n_train + n_cal :] <= upper)
            )
        )
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE
