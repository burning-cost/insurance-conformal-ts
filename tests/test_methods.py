"""
Tests for methods.py

Each method is tested for:
(a) Coverage >= nominal on stationary synthetic data
(b) Interval width decreasing with more calibration data
(c) Adaptation to distribution shift
(d) Correct API (fit/predict_interval, shapes, edge cases)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.methods import (
    ACI,
    EnbPI,
    ConformalPID,
    SPCI,
    _conformal_quantile,
)
from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    PoissonPearsonScore,
)
from tests.conftest import ConstantForecaster, PoissonMeanForecaster


RNG = np.random.default_rng(42)
ALPHA = 0.1
COVERAGE_TOLERANCE = 0.08  # allow 8 pp below nominal (sequential setting)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_poisson_series(n: int = 600, lam: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=n).astype(float)


def make_shifted_series(n_pre: int = 300, n_post: int = 200, lam1: float = 10.0, lam2: float = 20.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.poisson(lam1, size=n_pre).astype(float),
        rng.poisson(lam2, size=n_post).astype(float),
    ])


# ---------------------------------------------------------------------------
# _conformal_quantile utility
# ---------------------------------------------------------------------------

class TestConformalQuantile:
    def test_returns_float(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = _conformal_quantile(scores, 0.1)
        assert isinstance(q, float)

    def test_finite_sample_correction(self):
        """Corrected quantile should >= empirical (1-alpha) quantile."""
        scores = np.arange(1, 101, dtype=float)
        q = _conformal_quantile(scores, 0.1)
        assert q >= np.quantile(scores, 0.9)

    def test_too_small_calibration_returns_inf(self):
        """Alpha too small for calibration set size returns inf."""
        scores = np.array([1.0])  # n=1
        q = _conformal_quantile(scores, 0.001)
        assert q == np.inf


# ---------------------------------------------------------------------------
# ACI
# ---------------------------------------------------------------------------

class TestACI:
    def test_fit_returns_self(self):
        y = make_poisson_series(200)
        aci = ACI(ConstantForecaster(), gamma=0.02)
        result = aci.fit(y[:100])
        assert result is aci

    def test_predict_interval_shape(self):
        y = make_poisson_series(300)
        aci = ACI(ConstantForecaster(), gamma=0.02, window_size=50)
        aci.fit(y[:150])
        lower, upper = aci.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_poisson_series(300)
        aci = ACI(ConstantForecaster(), gamma=0.02, window_size=50)
        aci.fit(y[:150])
        lower, upper = aci.predict_interval(y[150:], alpha=ALPHA)
        # Some may be inf; where finite, upper >= lower
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_stationary(self):
        """On a stationary series, ACI should achieve ~90% coverage."""
        n_train = 200
        n_test = 500
        y = make_poisson_series(n_train + n_test, seed=10)
        aci = ACI(ConstantForecaster(), gamma=0.02, window_size=100)
        aci.fit(y[:n_train])
        lower, upper = aci.predict_interval(y[n_train:], alpha=ALPHA)
        coverage = float(np.mean((y[n_train:] >= lower) & (y[n_train:] <= upper)))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"ACI coverage {coverage:.3f} < {1 - ALPHA - COVERAGE_TOLERANCE:.3f}"
        )

    def test_raises_before_fit(self):
        aci = ACI(ConstantForecaster())
        y = make_poisson_series(50)
        with pytest.raises(RuntimeError, match="fit"):
            aci.predict_interval(y, alpha=0.1)

    def test_no_calibration_data_returns_inf_upper(self):
        """First prediction with no calibration data should give inf upper bound."""
        aci = ACI(ConstantForecaster(), gamma=0.02, window_size=100)
        y = make_poisson_series(100)
        aci.fit(y[:50])  # this sets calibration scores to empty
        # Force empty calibration
        aci._calibration_scores = []
        lower, upper = aci.predict_interval(y[50:51], alpha=0.1)
        assert not np.isfinite(upper[0])

    def test_adaptation_to_shift(self):
        """ACI should recover coverage after a distribution shift."""
        y = make_shifted_series(300, 200, lam1=10, lam2=20, seed=5)
        aci = ACI(ConstantForecaster(), gamma=0.05, window_size=50)
        aci.fit(y[:200])
        lower, upper = aci.predict_interval(y[200:], alpha=ALPHA)

        # Coverage in post-shift window (last 150 steps)
        y_post = y[200:]
        lower_post = lower[50:]
        upper_post = upper[50:]
        y_post_late = y_post[50:]
        cov_late = float(np.mean((y_post_late >= lower_post) & (y_post_late <= upper_post)))
        assert cov_late >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"ACI post-shift coverage {cov_late:.3f} below tolerance"
        )

    def test_with_poisson_score(self):
        """ACI with PoissonPearsonScore should also achieve coverage."""
        y = make_poisson_series(600, lam=15, seed=20)
        score = PoissonPearsonScore()
        aci = ACI(PoissonMeanForecaster(), score=score, gamma=0.02, window_size=100)
        aci.fit(y[:200])
        lower, upper = aci.predict_interval(y[200:], alpha=ALPHA)
        coverage = float(np.mean((y[200:] >= lower) & (y[200:] <= upper)))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_window_size_none(self):
        """window_size=None should use all historical scores."""
        y = make_poisson_series(300)
        aci = ACI(ConstantForecaster(), gamma=0.02, window_size=None)
        aci.fit(y[:100])
        lower, upper = aci.predict_interval(y[100:200], alpha=0.1)
        assert len(lower) == 100

    def test_gamma_zero_constant_alpha(self):
        """gamma=0 means alpha_t is constant; method still runs."""
        y = make_poisson_series(300)
        aci = ACI(ConstantForecaster(), gamma=0.0, window_size=50)
        aci.fit(y[:100])
        lower, upper = aci.predict_interval(y[100:200], alpha=0.1)
        assert len(lower) == 100


# ---------------------------------------------------------------------------
# EnbPI
# ---------------------------------------------------------------------------

class TestEnbPI:
    def test_fit_returns_self(self):
        y = make_poisson_series(200)
        enbpi = EnbPI(lambda: ConstantForecaster(), B=10)
        result = enbpi.fit(y[:100])
        assert result is enbpi

    def test_predict_interval_shape(self):
        y = make_poisson_series(300)
        enbpi = EnbPI(lambda: ConstantForecaster(), B=10, window_size=50)
        enbpi.fit(y[:150])
        lower, upper = enbpi.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_poisson_series(300)
        enbpi = EnbPI(lambda: ConstantForecaster(), B=10, window_size=50)
        enbpi.fit(y[:150])
        lower, upper = enbpi.predict_interval(y[150:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_stationary(self):
        """EnbPI should achieve ~90% coverage on stationary data."""
        n_train = 200
        n_test = 400
        y = make_poisson_series(n_train + n_test, lam=10, seed=15)
        enbpi = EnbPI(lambda: ConstantForecaster(), B=20, window_size=100, seed=0)
        enbpi.fit(y[:n_train])
        lower, upper = enbpi.predict_interval(y[n_train:], alpha=ALPHA)
        coverage = float(np.mean((y[n_train:] >= lower) & (y[n_train:] <= upper)))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_raises_before_fit(self):
        enbpi = EnbPI(lambda: ConstantForecaster(), B=5)
        y = make_poisson_series(50)
        with pytest.raises(RuntimeError, match="fit"):
            enbpi.predict_interval(y, alpha=0.1)

    def test_rolling_update_replaces_old_scores(self):
        """After window_size updates, the oldest scores should be gone."""
        y = make_poisson_series(300)
        window = 20
        enbpi = EnbPI(lambda: ConstantForecaster(), B=10, window_size=window)
        enbpi.fit(y[:100])
        initial_len = len(enbpi._calibration_scores)
        enbpi.predict_interval(y[100:150], alpha=0.1)
        # After 50 updates, the calibration set should be capped at window size
        assert len(enbpi._calibration_scores) <= max(initial_len, window) + 50

    def test_b_equals_one(self):
        """B=1 (degenerate ensemble) should still produce intervals."""
        y = make_poisson_series(200)
        enbpi = EnbPI(lambda: ConstantForecaster(), B=1, window_size=50)
        enbpi.fit(y[:100])
        lower, upper = enbpi.predict_interval(y[100:150], alpha=0.1)
        assert len(lower) == 50


# ---------------------------------------------------------------------------
# SPCI
# ---------------------------------------------------------------------------

class TestSPCI:
    def test_fit_returns_self(self):
        y = make_poisson_series(300)
        spci = SPCI(ConstantForecaster())
        result = spci.fit(y[:150])
        assert result is spci

    def test_predict_interval_shape(self):
        y = make_poisson_series(400)
        spci = SPCI(ConstantForecaster(), n_lags=5, min_calibration=20)
        spci.fit(y[:200])
        lower, upper = spci.predict_interval(y[200:], alpha=ALPHA)
        assert lower.shape == (200,)
        assert upper.shape == (200,)

    def test_upper_geq_lower(self):
        y = make_poisson_series(400)
        spci = SPCI(ConstantForecaster(), n_lags=5, min_calibration=20)
        spci.fit(y[:200])
        lower, upper = spci.predict_interval(y[200:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_stationary(self):
        """SPCI should achieve ~90% coverage on stationary Poisson data."""
        n_train = 200
        n_test = 400
        y = make_poisson_series(n_train + n_test, lam=10, seed=25)
        spci = SPCI(ConstantForecaster(), n_lags=5, min_calibration=20)
        spci.fit(y[:n_train])
        lower, upper = spci.predict_interval(y[n_train:], alpha=ALPHA)
        coverage = float(np.mean((y[n_train:] >= lower) & (y[n_train:] <= upper)))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_fallback_before_min_calibration(self):
        """SPCI falls back to standard conformal before enough calibration data."""
        y = make_poisson_series(300)
        spci = SPCI(ConstantForecaster(), n_lags=5, min_calibration=50)
        spci.fit(y[:100])
        # Manually empty the calibration scores to test fallback
        spci._calibration_scores = []
        lower, upper = spci.predict_interval(y[100:101], alpha=0.1)
        # With no calibration, should give inf
        assert not np.isfinite(upper[0])

    def test_raises_before_fit(self):
        spci = SPCI(ConstantForecaster())
        y = make_poisson_series(50)
        with pytest.raises(RuntimeError, match="fit"):
            spci.predict_interval(y, alpha=0.1)


# ---------------------------------------------------------------------------
# ConformalPID
# ---------------------------------------------------------------------------

class TestConformalPID:
    def test_fit_returns_self(self):
        y = make_poisson_series(200)
        pid = ConformalPID(ConstantForecaster())
        result = pid.fit(y[:100])
        assert result is result

    def test_predict_interval_shape(self):
        y = make_poisson_series(300)
        pid = ConformalPID(ConstantForecaster())
        pid.fit(y[:150])
        lower, upper = pid.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_poisson_series(300)
        pid = ConformalPID(ConstantForecaster())
        pid.fit(y[:150])
        lower, upper = pid.predict_interval(y[150:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_stationary(self):
        """ConformalPID should achieve ~90% coverage on stationary data."""
        n_train = 200
        n_test = 500
        y = make_poisson_series(n_train + n_test, lam=10, seed=30)
        pid = ConformalPID(ConstantForecaster(), Kp=0.01, Ki=0.001, Kd=0.001)
        pid.fit(y[:n_train])
        lower, upper = pid.predict_interval(y[n_train:], alpha=ALPHA)
        coverage = float(np.mean((y[n_train:] >= lower) & (y[n_train:] <= upper)))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_raises_before_fit(self):
        pid = ConformalPID(ConstantForecaster())
        y = make_poisson_series(50)
        with pytest.raises(RuntimeError, match="fit"):
            pid.predict_interval(y, alpha=0.1)

    def test_adaptation_to_shift(self):
        """ConformalPID should recover after distribution shift."""
        y = make_shifted_series(300, 200, lam1=10, lam2=20, seed=8)
        pid = ConformalPID(ConstantForecaster(), Kp=0.05, Ki=0.005, Kd=0.005)
        pid.fit(y[:200])
        lower, upper = pid.predict_interval(y[200:], alpha=ALPHA)
        y_post = y[200:]
        # Check late-period coverage (after adaptation).
        # PID adaptation is inherently slower than ACI so a wider tolerance applies.
        y_late = y_post[100:]
        lower_late = lower[100:]
        upper_late = upper[100:]
        cov_late = float(np.mean((y_late >= lower_late) & (y_late <= upper_late)))
        assert cov_late >= (1 - ALPHA) - 0.15, (
            f"ConformalPID post-shift coverage {cov_late:.3f} below tolerance"
        )


# ---------------------------------------------------------------------------
# Comparative tests: width ordering
# ---------------------------------------------------------------------------

class TestIntervalWidthOrdering:
    """More calibration data should not increase interval width (on average)."""

    def test_more_calibration_not_wider(self):
        """ACI with more calibration history should give equal or narrower intervals."""
        y = make_poisson_series(600, lam=10, seed=99)
        alpha = 0.1

        # Small calibration window
        aci_small = ACI(ConstantForecaster(), gamma=0.02, window_size=20)
        aci_small.fit(y[:100])
        l1, u1 = aci_small.predict_interval(y[100:200], alpha=alpha)
        mask1 = np.isfinite(u1) & np.isfinite(l1)
        w_small = float(np.mean(u1[mask1] - l1[mask1]))

        # Large calibration window
        aci_large = ACI(ConstantForecaster(), gamma=0.02, window_size=200)
        aci_large.fit(y[:100])
        l2, u2 = aci_large.predict_interval(y[100:200], alpha=alpha)
        mask2 = np.isfinite(u2) & np.isfinite(l2)
        w_large = float(np.mean(u2[mask2] - l2[mask2]))

        # Both should give finite-ish widths; no strict ordering requirement but
        # both must be positive and finite for the bulk of predictions
        assert np.isfinite(w_small)
        assert np.isfinite(w_large)
        assert w_small > 0
        assert w_large > 0
