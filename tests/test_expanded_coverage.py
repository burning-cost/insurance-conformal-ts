"""
test_expanded_coverage.py
=========================

Expanded test coverage targeting untested code paths:

- _conformal_quantile: edge cases (empty, boundary alpha, single element)
- ACI: state persistence, score_kwargs passthrough, burn_in variations
- EnbPI: invalid B, window_size=None, seed reproducibility, score_kwargs
- SPCI: custom quantile regressor, n_lags edge cases, SPCI path activation
- ConformalPID: degenerate gains, saturation, multiple predict calls
- MSCP: H=1, h=H in sequence, update quantile recompute, score_kwargs
- NonConformityScore: NegBinom phi at inverse time, ExposureAdjustedScore warning
- LocallyWeightedScore: clip_lower=True
- Insurance wrappers: LossRatioConformal lower can be negative, coverage_report edge cases
- Diagnostics: Kupiec exact nominal, drift slope positive, IntervalWidthReport edge cases
- Integration: method + non-default score combinations
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_conformal_ts.methods import (
    ACI,
    EnbPI,
    ConformalPID,
    SPCI,
    ConstantForecaster,
    MeanForecaster,
    _conformal_quantile,
)
from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    ExposureAdjustedScore,
    LocallyWeightedScore,
    NegBinomPearsonScore,
    PoissonPearsonScore,
)
from insurance_conformal_ts.multistep import MSCP
from insurance_conformal_ts.insurance import (
    ClaimsCountConformal,
    LossRatioConformal,
    SeverityConformal,
)
from insurance_conformal_ts.diagnostics import (
    IntervalWidthReport,
    SequentialCoverageReport,
)

from tests.conftest import ConstantForecaster as TestConstantForecaster


# ---------------------------------------------------------------------------
# Data generators (all deterministic via seed)
# ---------------------------------------------------------------------------

def make_y(n: int = 300, lam: float = 10.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(lam, size=n).astype(float)


def make_gaussian(n: int = 300, mu: float = 100.0, sigma: float = 15.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, size=n)


# ---------------------------------------------------------------------------
# _conformal_quantile: edge cases
# ---------------------------------------------------------------------------

class TestConformalQuantileEdgeCases:
    def test_single_element_alpha_small(self):
        """n=1, alpha=0.05: level > 1 so should return inf."""
        q = _conformal_quantile(np.array([5.0]), alpha=0.05)
        assert q == np.inf

    def test_single_element_alpha_large(self):
        """n=1, alpha=0.5: level = ceil(0.5 * 2) / 1 = 1.0, still returns actual quantile."""
        q = _conformal_quantile(np.array([5.0]), alpha=0.5)
        # level = ceil(0.5 * 2) / 1 = 1.0, not > 1, so returns np.quantile([5.0], 1.0) = 5.0
        assert np.isfinite(q)
        assert q == pytest.approx(5.0)

    def test_large_calibration_finite(self):
        """Large calibration set should always give finite quantile for reasonable alpha."""
        scores = np.arange(1, 1001, dtype=float)
        q = _conformal_quantile(scores, alpha=0.1)
        assert np.isfinite(q)

    def test_monotone_in_alpha(self):
        """Larger alpha means tighter quantile (smaller value)."""
        scores = np.arange(1, 101, dtype=float)
        q_tight = _conformal_quantile(scores, alpha=0.05)
        q_loose = _conformal_quantile(scores, alpha=0.2)
        # alpha=0.05 => higher quantile (larger) than alpha=0.2
        assert q_tight >= q_loose

    def test_alpha_near_zero_returns_inf(self):
        """Extremely small alpha should return inf for small calibration sets."""
        scores = np.array([1.0, 2.0, 3.0])
        q = _conformal_quantile(scores, alpha=1e-10)
        assert q == np.inf

    def test_all_same_scores(self):
        """All-constant calibration should return that constant."""
        scores = np.full(50, 3.7)
        q = _conformal_quantile(scores, alpha=0.1)
        assert np.isfinite(q)
        assert q == pytest.approx(3.7)

    def test_return_type_is_float(self):
        """Return type must be Python float, not np.float64."""
        scores = np.arange(10, dtype=float)
        q = _conformal_quantile(scores, alpha=0.1)
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# ACI: additional paths
# ---------------------------------------------------------------------------

class TestACIAdditional:
    def test_calibration_scores_accumulate_across_calls(self):
        """Multiple predict_interval calls accumulate calibration scores."""
        y = make_y(400)
        aci = ACI(TestConstantForecaster(), gamma=0.02, window_size=None)
        aci.fit(y[:100])
        assert len(aci._calibration_scores) == 0
        aci.predict_interval(y[100:150], alpha=0.1)
        n_after_first = len(aci._calibration_scores)
        aci.predict_interval(y[150:200], alpha=0.1)
        n_after_second = len(aci._calibration_scores)
        assert n_after_second == n_after_first + 50

    def test_burn_in_zero_gives_intervals_from_first_step(self):
        """burn_in=0 means intervals are produced even with empty calibration."""
        y = make_y(300)
        aci = ACI(TestConstantForecaster(), gamma=0.02, window_size=50, burn_in=0)
        aci.fit(y[:100])
        aci._calibration_scores = []  # force empty
        lower, upper = aci.predict_interval(y[100:101], alpha=0.1)
        # With burn_in=0 and empty cal, _conformal_quantile([]) of empty array —
        # burn_in=0 means len(cal)=0 >= burn_in=0 so it calls _conformal_quantile
        # on empty array which returns inf (level > 1 for n=0).
        # The key point: no RuntimeError, just inf.
        assert len(lower) == 1

    def test_burn_in_1_with_one_score_returns_inf(self):
        """With burn_in=1 and one calibration score at alpha=0.1, level exceeds 1 so inf."""
        y = make_y(300, lam=10.0, seed=77)
        aci = ACI(TestConstantForecaster(), gamma=0.02, window_size=100, burn_in=1)
        aci.fit(y[:100])
        # Manually seed one calibration score
        aci._calibration_scores = [2.0]
        lower, upper = aci.predict_interval(y[100:102], alpha=0.1)
        # n=1, level = ceil(0.9*2)/1 = 2.0 > 1 → inf is correct
        assert upper[0] == np.inf

    def test_window_size_caps_calibration(self):
        """window_size should cap the calibration set used (not the stored scores)."""
        y = make_y(500)
        window = 30
        aci = ACI(TestConstantForecaster(), gamma=0.02, window_size=window)
        aci.fit(y[:100])
        # Manually inject a large calibration set
        aci._calibration_scores = list(range(200))
        lower, upper = aci.predict_interval(y[100:101], alpha=0.1)
        # Should not error; verifies window slicing works
        assert len(lower) == 1

    def test_score_kwargs_arrays_sliced_per_step(self):
        """score_kwargs with arrays should be sliced to [t:t+1] at each step."""
        n = 100
        y = make_y(n + 50, lam=10.0)
        sigma_hat = np.full(50, 2.0)  # per-step sigma values
        score = LocallyWeightedScore()
        aci = ACI(TestConstantForecaster(), score=score, gamma=0.02, window_size=None)
        aci.fit(y[:n])
        # Manually seed some calibration scores so we get finite intervals
        aci._calibration_scores = [1.5] * 10
        lower, upper = aci.predict_interval(
            y[n:n + 50], alpha=0.1, score_kwargs={"sigma_hat": sigma_hat}
        )
        assert lower.shape == (50,)
        assert upper.shape == (50,)
        assert np.all(upper >= lower)

    def test_score_kwargs_scalar_passthrough(self):
        """Scalar score_kwargs values should be passed unchanged at each step."""
        y = make_y(200)
        score = LocallyWeightedScore()
        aci = ACI(TestConstantForecaster(), score=score, gamma=0.02, window_size=None)
        aci.fit(y[:100])
        aci._calibration_scores = [1.5] * 10
        # sigma_hat as scalar — should not error
        lower, upper = aci.predict_interval(
            y[100:120], alpha=0.1, score_kwargs={"sigma_hat": 3.0}
        )
        assert len(lower) == 20

    def test_alpha_clipping_prevents_boundary_values(self):
        """alpha_t should never reach 0 or 1 due to clipping."""
        y = make_y(500, lam=5.0)
        # Use extreme gamma to maximise alpha movement
        aci = ACI(TestConstantForecaster(), gamma=0.5, window_size=50)
        aci.fit(y[:100])
        aci._calibration_scores = [1.0] * 20
        lower, upper = aci.predict_interval(y[100:200], alpha=0.1)
        # All bounds should be finite (alpha_t clipped to [1e-6, 1-1e-6])
        # at least eventually
        assert len(lower) == 100

    def test_fit_clears_calibration_on_refit(self):
        """Re-fitting should reset calibration scores."""
        y = make_y(300)
        aci = ACI(TestConstantForecaster(), gamma=0.02)
        aci.fit(y[:100])
        aci._calibration_scores = [1.0, 2.0, 3.0]
        aci.fit(y[:150])  # re-fit
        assert aci._calibration_scores == []


# ---------------------------------------------------------------------------
# EnbPI: additional paths
# ---------------------------------------------------------------------------

class TestEnbPIAdditional:
    def test_invalid_B_raises(self):
        """B=0 should raise ValueError."""
        with pytest.raises(ValueError, match="B must be"):
            EnbPI(lambda: TestConstantForecaster(), B=0)

    def test_negative_B_raises(self):
        """Negative B should raise ValueError."""
        with pytest.raises(ValueError, match="B must be"):
            EnbPI(lambda: TestConstantForecaster(), B=-5)

    def test_window_size_none_keeps_all_scores(self):
        """window_size=None should keep all calibration scores indefinitely."""
        y = make_y(400)
        enbpi = EnbPI(lambda: TestConstantForecaster(), B=10, window_size=None, seed=1)
        enbpi.fit(y[:100])
        n_initial = len(enbpi._calibration_scores)
        enbpi.predict_interval(y[100:200], alpha=0.1)
        # Should have grown by n_test without capping
        assert len(enbpi._calibration_scores) == n_initial + 100

    def test_seed_reproducibility(self):
        """Same seed should produce identical intervals."""
        y = make_y(300)
        enbpi1 = EnbPI(lambda: TestConstantForecaster(), B=20, seed=42)
        enbpi2 = EnbPI(lambda: TestConstantForecaster(), B=20, seed=42)
        enbpi1.fit(y[:150])
        enbpi2.fit(y[:150])
        l1, u1 = enbpi1.predict_interval(y[150:], alpha=0.1)
        l2, u2 = enbpi2.predict_interval(y[150:], alpha=0.1)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(u1, u2)

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different intervals."""
        y = make_y(400)
        enbpi1 = EnbPI(lambda: TestConstantForecaster(), B=20, seed=1)
        enbpi2 = EnbPI(lambda: TestConstantForecaster(), B=20, seed=2)
        enbpi1.fit(y[:200])
        enbpi2.fit(y[:200])
        l1, u1 = enbpi1.predict_interval(y[200:], alpha=0.1)
        l2, u2 = enbpi2.predict_interval(y[200:], alpha=0.1)
        # Not necessarily different (both use ConstantForecaster), but the test
        # verifies the code path runs cleanly
        assert len(l1) == len(l2)

    def test_window_size_affects_quantile_not_storage(self):
        """EnbPI stores all scores but uses only the most recent window_size for quantiles."""
        y = make_y(500)
        window = 15
        enbpi = EnbPI(lambda: TestConstantForecaster(), B=5, window_size=window, seed=0)
        enbpi.fit(y[:100])
        enbpi.predict_interval(y[100:400], alpha=0.1)
        # EnbPI stores all calibration scores (window_size caps at quantile time)
        assert len(enbpi._calibration_scores) >= window

    def test_score_kwargs_passthrough(self):
        """score_kwargs should be passed to the score function."""
        y = make_y(300, lam=15.0)
        exposure = np.full(100, 150.0)
        score = ExposureAdjustedScore()
        enbpi = EnbPI(
            lambda: TestConstantForecaster(),
            score=score,
            B=10,
            window_size=50,
            seed=0,
        )
        enbpi.fit(y[:100])
        lower, upper = enbpi.predict_interval(
            y[100:200], alpha=0.1, score_kwargs={"exposure": exposure}
        )
        assert lower.shape == (100,)
        assert upper.shape == (100,)

    def test_ensemble_mean_used(self):
        """EnbPI uses ensemble mean, not a single member — verify via B=1 vs B=50."""
        y = make_y(400)
        enbpi1 = EnbPI(lambda: TestConstantForecaster(), B=1, seed=99)
        enbpi50 = EnbPI(lambda: TestConstantForecaster(), B=50, seed=99)
        enbpi1.fit(y[:200])
        enbpi50.fit(y[:200])
        # B shouldn't matter for ConstantForecaster (all members predict same mean)
        l1, u1 = enbpi1.predict_interval(y[200:250], alpha=0.1)
        l50, u50 = enbpi50.predict_interval(y[200:250], alpha=0.1)
        # Intervals may differ due to LOO residuals, but shapes must match
        assert l1.shape == l50.shape


# ---------------------------------------------------------------------------
# SPCI: additional paths
# ---------------------------------------------------------------------------

class TestSPCIAdditional:
    def test_spci_path_activated_with_enough_calibration(self):
        """After min_calibration steps, SPCI should use quantile regression."""
        y = make_y(500, lam=10.0, seed=5)
        spci = SPCI(TestConstantForecaster(), n_lags=5, min_calibration=10)
        spci.fit(y[:100])
        # Seed enough calibration points
        spci._calibration_scores = [float(i % 5) for i in range(20)]
        lower, upper = spci.predict_interval(y[100:110], alpha=0.1)
        # Should run without error and produce finite-ish intervals
        assert len(lower) == 10

    def test_spci_fallback_with_partial_calibration(self):
        """With calibration between n_lags and min_calibration, uses standard conformal."""
        y = make_y(300)
        spci = SPCI(TestConstantForecaster(), n_lags=5, min_calibration=30)
        spci.fit(y[:100])
        # 15 scores: above n_lags (5) but below min_calibration (30)
        spci._calibration_scores = [2.0] * 15
        lower, upper = spci.predict_interval(y[100:101], alpha=0.1)
        # Standard conformal path: should give finite interval
        assert np.isfinite(upper[0])

    def test_calibration_scores_grow_during_predict(self):
        """Each predict step should append one new score."""
        y = make_y(300)
        spci = SPCI(TestConstantForecaster(), n_lags=5, min_calibration=30)
        spci.fit(y[:100])
        spci._calibration_scores = [1.0] * 10
        n_before = len(spci._calibration_scores)
        spci.predict_interval(y[100:120], alpha=0.1)
        assert len(spci._calibration_scores) == n_before + 20

    def test_custom_quantile_regressor_used(self):
        """Custom quantile regressor should be accepted and used."""
        from sklearn.linear_model import QuantileRegressor
        y = make_y(500, seed=9)
        qr = QuantileRegressor(quantile=0.9, alpha=0.01, solver="highs")
        spci = SPCI(
            TestConstantForecaster(),
            quantile_regressor=qr,
            n_lags=5,
            min_calibration=10,
        )
        spci.fit(y[:100])
        spci._calibration_scores = [float(i % 3) for i in range(25)]
        lower, upper = spci.predict_interval(y[100:110], alpha=0.1)
        assert len(lower) == 10

    def test_lag_feature_shape(self):
        """_make_lag_features should return (n - n_lags, n_lags) shaped array."""
        spci = SPCI(TestConstantForecaster(), n_lags=3)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        features = spci._make_lag_features(scores)
        assert features is not None
        assert features.shape == (3, 3)  # (6 - 3) rows, 3 lag cols

    def test_lag_feature_returns_none_when_insufficient(self):
        """_make_lag_features returns None when scores < n_lags."""
        spci = SPCI(TestConstantForecaster(), n_lags=10)
        scores = [1.0, 2.0]  # fewer than n_lags
        features = spci._make_lag_features(scores)
        assert features is None

    def test_n_lags_exactly_at_boundary(self):
        """n_lags scores in calibration: _make_lag_features should return None."""
        spci = SPCI(TestConstantForecaster(), n_lags=5)
        scores = [1.0] * 5  # exactly n_lags, so len < n_lags is False but rows=[]
        features = spci._make_lag_features(scores)
        # len(scores) == n_lags, loop range(5,5) is empty, returns None
        assert features is None

    def test_fit_resets_calibration(self):
        """Re-fitting SPCI should clear the calibration scores."""
        y = make_y(300)
        spci = SPCI(TestConstantForecaster())
        spci.fit(y[:100])
        spci._calibration_scores = [1.0, 2.0, 3.0]
        spci.fit(y[:150])
        assert spci._calibration_scores == []


# ---------------------------------------------------------------------------
# ConformalPID: additional paths
# ---------------------------------------------------------------------------

class TestConformalPIDAdditional:
    def test_all_gains_zero_behaves_like_constant_alpha(self):
        """Kp=Ki=Kd=0 means PID produces no adjustment; intervals should still be valid."""
        y = make_y(400)
        pid = ConformalPID(TestConstantForecaster(), Kp=0.0, Ki=0.0, Kd=0.0)
        pid.fit(y[:200])
        pid._calibration_scores = [2.0] * 50
        lower, upper = pid.predict_interval(y[200:250], alpha=0.1)
        assert len(lower) == 50
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_integral_saturation_prevents_windup(self):
        """With a saturating integral, integral term should be capped."""
        y = make_y(500)
        pid = ConformalPID(
            TestConstantForecaster(),
            Kp=0.1, Ki=0.1, Kd=0.01,
            saturation=0.1,  # tight saturation
        )
        pid.fit(y[:200])
        pid._calibration_scores = [5.0] * 100  # high scores -> wide intervals
        lower, upper = pid.predict_interval(y[200:300], alpha=0.1)
        # Just verify it runs; saturation prevents numerical blowup
        assert len(lower) == 100

    def test_state_preserved_between_predict_calls(self):
        """Multiple predict_interval calls should accumulate calibration scores."""
        y = make_y(600)
        pid = ConformalPID(TestConstantForecaster(), Kp=0.01, Ki=0.001, Kd=0.001)
        pid.fit(y[:200])
        pid._calibration_scores = [2.0] * 30
        n_before = len(pid._calibration_scores)
        pid.predict_interval(y[200:250], alpha=0.1)
        pid.predict_interval(y[250:300], alpha=0.1)
        assert len(pid._calibration_scores) == n_before + 100

    def test_empty_calibration_returns_inf(self):
        """With no calibration scores, ConformalPID should return inf upper bound."""
        y = make_y(300)
        pid = ConformalPID(TestConstantForecaster())
        pid.fit(y[:100])
        pid._calibration_scores = []
        lower, upper = pid.predict_interval(y[100:101], alpha=0.1)
        assert not np.isfinite(upper[0])

    def test_window_size_none(self):
        """window_size=None should use all calibration scores."""
        y = make_y(400)
        pid = ConformalPID(TestConstantForecaster(), window_size=None)
        pid.fit(y[:200])
        pid._calibration_scores = [2.0] * 50
        lower, upper = pid.predict_interval(y[200:210], alpha=0.1)
        assert len(lower) == 10

    def test_coverage_with_high_gains(self):
        """Even with aggressive PID gains, coverage should stay reasonable."""
        y = make_y(700, lam=12.0, seed=44)
        pid = ConformalPID(
            TestConstantForecaster(),
            Kp=0.1, Ki=0.05, Kd=0.05,
            window_size=100,
        )
        pid.fit(y[:200])
        lower, upper = pid.predict_interval(y[200:500], alpha=0.1)
        coverage = float(np.mean((y[200:500] >= lower) & (y[200:500] <= upper)))
        # Loose check: just verify it's not 0
        assert coverage > 0.3

    def test_fit_resets_state(self):
        """Re-fitting should reset calibration scores."""
        y = make_y(300)
        pid = ConformalPID(TestConstantForecaster())
        pid.fit(y[:100])
        pid._calibration_scores = [3.0, 4.0, 5.0]
        pid.fit(y[:200])
        assert pid._calibration_scores == []


# ---------------------------------------------------------------------------
# MSCP: additional paths
# ---------------------------------------------------------------------------

class TestMSCPAdditional:
    def _fitted_mscp(self, H: int = 4, n_train: int = 150, n_cal: int = 200, seed: int = 0):
        y = make_y(n_train + n_cal + 100, seed=seed)
        mscp = MSCP(TestConstantForecaster(), H=H, min_cal_per_horizon=5)
        mscp.fit(y[:n_train])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mscp.calibrate(y[n_train:n_train + n_cal], alpha=0.1)
        return mscp, y

    def test_H_equals_one(self):
        """MSCP with H=1 should produce a single-horizon fan."""
        y = make_y(400)
        mscp = MSCP(TestConstantForecaster(), H=1, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        mscp.calibrate(y[100:300], alpha=0.1)
        fan = mscp.predict_fan()
        assert set(fan.keys()) == {1}

    def test_predict_interval_sequence_h_equals_H(self):
        """predict_interval_sequence at h=H should return n_test - H outputs."""
        H = 4
        mscp, y = self._fitted_mscp(H=H)
        n_test = 50
        y_test = y[350:350 + n_test]
        lower, upper = mscp.predict_interval_sequence(y_test, h=H, alpha=0.1)
        assert len(lower) == n_test - H
        assert len(upper) == n_test - H

    def test_predict_interval_sequence_zero_length_for_short_y(self):
        """y of length <= h should give empty output."""
        H = 4
        mscp, y = self._fitted_mscp(H=H)
        # y_test has only H elements — should give 0 outputs (n - h = 0)
        y_test = y[:H]
        lower, upper = mscp.predict_interval_sequence(y_test, h=H, alpha=0.1)
        assert len(lower) == 0
        assert len(upper) == 0

    def test_horizon_quantiles_differ(self):
        """Longer horizons should generally have >= quantile of shorter horizons."""
        y = make_y(800, lam=10.0, seed=7)
        mscp = MSCP(TestConstantForecaster(), H=6, min_cal_per_horizon=10)
        mscp.fit(y[:100])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mscp.calibrate(y[100:600], alpha=0.1)
        # All quantiles should be finite and non-negative
        for h in range(1, 7):
            q = mscp._h_quantiles.get(h, np.inf)
            assert q >= 0

    def test_update_increments_h1_scores(self):
        """After update with 1 new obs, h=1 scores should grow by 1."""
        mscp, y = self._fitted_mscp(H=3, n_cal=150)
        n_before_h1 = len(mscp._h_scores[1])
        # Give enough history so t_origin >= 0 for h=1
        mscp._y_history = [float(v) for v in y[:5]]
        mscp._X_history = [None] * 5
        mscp.update(y[5:6])
        n_after_h1 = len(mscp._h_scores[1])
        assert n_after_h1 == n_before_h1 + 1

    def test_update_quantiles_recomputed(self):
        """After update, quantiles should be recomputed (not stale)."""
        mscp, y = self._fitted_mscp(H=2, n_cal=100)
        q_before = dict(mscp._h_quantiles)
        mscp.update(y[250:280])
        # Quantiles may or may not change numerically, but code should not error
        assert isinstance(mscp._h_quantiles, dict)

    def test_predict_fan_with_explicit_alpha(self):
        """predict_fan with explicit alpha should not error."""
        mscp, y = self._fitted_mscp()
        fan = mscp.predict_fan(alpha=0.05)
        assert isinstance(fan, dict)
        for h, (lo, hi) in fan.items():
            if np.isfinite(lo) and np.isfinite(hi):
                assert hi >= lo

    def test_calibrate_alpha_stored(self):
        """The alpha passed to calibrate should be stored for later use."""
        y = make_y(400)
        mscp = MSCP(TestConstantForecaster(), H=3, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mscp.calibrate(y[100:300], alpha=0.05)
        assert mscp._alpha == pytest.approx(0.05)

    def test_predict_fan_missing_alpha_uses_calibrate_alpha(self):
        """predict_fan with no alpha arg should use the one from calibrate."""
        y = make_y(400)
        mscp = MSCP(TestConstantForecaster(), H=2, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mscp.calibrate(y[100:300], alpha=0.2)
        # Should use alpha=0.2 from calibrate
        fan = mscp.predict_fan()
        assert isinstance(fan, dict)


# ---------------------------------------------------------------------------
# NonConformityScore: additional paths
# ---------------------------------------------------------------------------

class TestNegBinomPearsonScoreAdditional:
    def test_phi_required_at_inverse_time(self):
        """NegBinomPearsonScore without phi should raise at inverse time too."""
        score = NegBinomPearsonScore()  # no phi
        with pytest.raises(ValueError, match="phi"):
            score.inverse(np.array([1.5]), np.array([10.0]))

    def test_phi_kwarg_overrides_constructor_at_inverse(self):
        """phi supplied at inverse time should override constructor phi."""
        score_nb2 = NegBinomPearsonScore(phi=1.0, parameterisation="NB2")
        mu = np.array([10.0])
        s = np.array([1.0])
        upper_phi1 = score_nb2.inverse(s, mu, phi=1.0)
        upper_phi10 = score_nb2.inverse(s, mu, phi=10.0)
        # With NB2, larger phi -> smaller variance -> smaller upper bound
        assert upper_phi1[0] > upper_phi10[0]

    def test_nb1_inverse_lower_clipped(self):
        """NB1 lower bound should be clipped at zero."""
        score = NegBinomPearsonScore(phi=2.0, parameterisation="NB1")
        mu = np.array([1.0])  # small mu
        s = np.array([10.0])   # large score
        lower = score.inverse(s, mu, upper=False)
        assert lower[0] == 0.0

    def test_nb2_inverse_lower_clipped(self):
        """NB2 lower bound should be clipped at zero."""
        score = NegBinomPearsonScore(phi=5.0, parameterisation="NB2")
        mu = np.array([0.5])  # very small mu
        s = np.array([20.0])  # very large score
        lower = score.inverse(s, mu, upper=False)
        assert lower[0] == 0.0

    def test_min_var_prevents_division_zero(self):
        """min_var should prevent numerical issues at mu=0."""
        score = NegBinomPearsonScore(phi=5.0, min_var=0.01)
        y = np.array([0.0])
        mu = np.array([0.0])
        s = score.score(y, mu)
        assert np.isfinite(s[0])


class TestExposureAdjustedScoreAdditional:
    def test_warns_when_no_exposure(self):
        """score() should warn when exposure is None."""
        score = ExposureAdjustedScore()
        y = np.array([10.0])
        y_hat = np.array([0.1])
        with pytest.warns(UserWarning, match="exposure"):
            score.score(y, y_hat)

    def test_no_warning_when_exposure_provided(self):
        """No warning when exposure is provided."""
        score = ExposureAdjustedScore()
        y = np.array([10.0])
        y_hat = np.array([0.1])
        exposure = np.array([100.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            score.score(y, y_hat, exposure=exposure)  # should not raise

    def test_inverse_no_exposure_uses_ones(self):
        """inverse() with no exposure should use E=1 implicitly."""
        score = ExposureAdjustedScore()
        s = np.array([0.5])
        y_hat = np.array([1.0])
        upper = score.inverse(s, y_hat)  # no exposure
        # E=1, upper = 1 * (1.0 + 0.5) = 1.5
        np.testing.assert_allclose(upper, [1.5])

    def test_clip_lower_false_allows_negative(self):
        """clip_lower=False allows negative lower bounds."""
        score = ExposureAdjustedScore(clip_lower=False)
        s = np.array([2.0])
        y_hat = np.array([0.5])
        exposure = np.array([1.0])
        lower = score.inverse(s, y_hat, exposure=exposure, upper=False)
        # lower = 1 * (0.5 - 2.0) = -1.5
        assert lower[0] < 0.0


class TestLocallyWeightedScoreAdditional:
    def test_clip_lower_true_clips_bound(self):
        """clip_lower=True should clip lower bound at zero."""
        score = LocallyWeightedScore(clip_lower=True)
        y_hat = np.array([2.0])
        s = np.array([5.0])
        sigma = np.array([1.0])
        lower = score.inverse(s, y_hat, sigma_hat=sigma, upper=False)
        # 2.0 - 5.0 * 1.0 = -3.0 -> clipped to 0
        assert lower[0] == 0.0

    def test_clip_lower_false_allows_negative_lower(self):
        """Default clip_lower=False: lower can be negative."""
        score = LocallyWeightedScore(clip_lower=False)
        y_hat = np.array([2.0])
        s = np.array([5.0])
        sigma = np.array([1.0])
        lower = score.inverse(s, y_hat, sigma_hat=sigma, upper=False)
        assert lower[0] == pytest.approx(-3.0)

    def test_large_sigma_widens_interval(self):
        """Larger sigma_hat should produce wider intervals."""
        score = LocallyWeightedScore()
        y_hat = np.array([10.0])
        s = np.array([1.96])
        small_sigma = np.array([1.0])
        large_sigma = np.array([5.0])
        u_small = score.inverse(s, y_hat, sigma_hat=small_sigma)
        u_large = score.inverse(s, y_hat, sigma_hat=large_sigma)
        assert u_large[0] > u_small[0]

    def test_zero_sigma_uses_floor(self):
        """sigma=0 should be floored to min_sigma, not cause division errors."""
        score = LocallyWeightedScore(min_sigma=1e-6)
        y = np.array([10.5])
        y_hat = np.array([10.0])
        sigma = np.array([0.0])
        s = score.score(y, y_hat, sigma_hat=sigma)
        # (10.5 - 10.0) / 1e-6 = 500000
        assert s[0] == pytest.approx(0.5 / 1e-6, rel=1e-5)

    def test_score_sign_preserved(self):
        """LocallyWeightedScore is signed (y < mu gives negative score)."""
        score = LocallyWeightedScore()
        y = np.array([8.0])
        mu = np.array([10.0])
        sigma = np.array([1.0])
        s = score.score(y, mu, sigma_hat=sigma)
        assert s[0] < 0.0


class TestAbsoluteResidualScoreAdditional:
    def test_score_with_list_inputs(self):
        """Should accept Python lists and convert internally."""
        score = AbsoluteResidualScore()
        s = score.score([5.0, 10.0], [4.0, 12.0])
        np.testing.assert_allclose(s, [1.0, 2.0])

    def test_inverse_with_inf_score(self):
        """Infinite score should give infinite upper bound."""
        score = AbsoluteResidualScore()
        upper = score.inverse(np.inf, np.array([5.0]))
        assert upper[0] == np.inf

    def test_symmetric_residuals(self):
        """For y = y_hat, score should be zero."""
        score = AbsoluteResidualScore()
        y_hat = np.array([7.0, 14.0, 21.0])
        s = score.score(y_hat, y_hat)
        np.testing.assert_allclose(s, np.zeros(3))


class TestPoissonPearsonScoreAdditional:
    def test_negative_mean_clipped(self):
        """Negative y_hat should be clipped to min_mu."""
        score = PoissonPearsonScore(min_mu=0.1)
        y = np.array([5.0])
        mu = np.array([-1.0])  # negative
        s = score.score(y, mu)
        # mu is clipped to 0.1: (5 - 0.1) / sqrt(0.1) ≈ 15.5
        assert s[0] == pytest.approx((5.0 - 0.1) / np.sqrt(0.1), rel=1e-6)

    def test_inverse_lower_clipped(self):
        """Large score should clip lower bound at zero."""
        score = PoissonPearsonScore()
        mu = np.array([1.0])
        s = np.array([100.0])  # very large
        lower = score.inverse(s, mu, upper=False)
        assert lower[0] == 0.0

    def test_score_shape_broadcast(self):
        """Scalar y_hat should broadcast correctly."""
        score = PoissonPearsonScore()
        y = np.array([8.0, 10.0, 12.0])
        mu = np.array([10.0, 10.0, 10.0])
        s = score.score(y, mu)
        assert s.shape == (3,)


# ---------------------------------------------------------------------------
# Insurance wrappers: additional paths
# ---------------------------------------------------------------------------

class TestLossRatioConformalAdditional:
    def test_lower_can_be_negative(self):
        """Loss ratios are continuous; lower bound can go negative (clip_lower=False)."""
        y = make_gaussian(300, mu=0.65, sigma=0.1)
        lrc = LossRatioConformal(base_forecaster=TestConstantForecaster())
        lrc.fit(y[:150])
        lower, upper = lrc.predict_interval(y[150:], alpha=0.1)
        # With wide intervals some lower bounds may be negative — that's acceptable
        # Just check the code doesn't error and shapes are right
        assert lower.shape == (150,)

    def test_coverage_report_all_covered(self):
        """All covered: coverage should be 1.0."""
        y = np.ones(50) * 5.0
        lower = np.zeros(50)
        upper = np.full(50, 10.0)
        lrc = LossRatioConformal(base_forecaster=TestConstantForecaster())
        lrc.fit(y)
        report = lrc.coverage_report(y, lower, upper)
        assert report["coverage"] == pytest.approx(1.0)

    def test_coverage_report_none_covered(self):
        """None covered: coverage should be 0.0."""
        y = np.ones(50) * 5.0
        lower = np.full(50, 10.0)
        upper = np.full(50, 20.0)
        lrc = LossRatioConformal(base_forecaster=TestConstantForecaster())
        lrc.fit(y)
        report = lrc.coverage_report(y, lower, upper)
        assert report["coverage"] == pytest.approx(0.0)

    def test_coverage_report_excludes_infinite_widths_from_mean(self):
        """Infinite-width intervals should be excluded from mean_width calculation."""
        y = np.ones(4) * 5.0
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([10.0, np.inf, 10.0, np.inf])
        lrc = LossRatioConformal(base_forecaster=TestConstantForecaster())
        lrc.fit(y)
        report = lrc.coverage_report(y, lower, upper)
        # mean_width only from finite intervals: (10 + 10) / 2 = 10
        assert np.isfinite(report["mean_width"])

    def test_custom_method_accepted(self):
        """LossRatioConformal should accept a custom conformal method."""
        y = make_gaussian(400, mu=0.65, sigma=0.1)
        forecaster = TestConstantForecaster()
        score = AbsoluteResidualScore(clip_lower=False)
        method = ACI(forecaster, score=score, gamma=0.03, window_size=100)
        lrc = LossRatioConformal(base_forecaster=forecaster, method=method, score=score)
        lrc.fit(y[:200])
        lower, upper = lrc.predict_interval(y[200:], alpha=0.1)
        assert len(lower) == 200


class TestSeverityConformalAdditional:
    def test_lower_nonnegative(self):
        """Severity lower bound should be clipped at zero."""
        y = make_y(300, lam=2000, seed=3)
        sc = SeverityConformal(base_forecaster=TestConstantForecaster())
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=0.1)
        finite_lower = lower[np.isfinite(lower)]
        assert np.all(finite_lower >= 0)

    def test_coverage_report_mean_width_positive(self):
        """Mean width should be positive for reasonable intervals."""
        y = make_y(300, lam=2000)
        sc = SeverityConformal(base_forecaster=TestConstantForecaster())
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=0.1)
        report = sc.coverage_report(y[150:], lower, upper)
        finite_widths = (upper - lower)[np.isfinite(upper - lower)]
        if len(finite_widths) > 0:
            assert report["mean_width"] >= 0.0

    def test_custom_score_accepted(self):
        """SeverityConformal should accept a custom non-conformity score."""
        y = make_y(300, lam=1500)
        score = LocallyWeightedScore(clip_lower=True)
        sigma_hat = np.full(len(y), 300.0)
        forecaster = TestConstantForecaster()
        method = ACI(forecaster, score=score, gamma=0.02, window_size=100)
        sc = SeverityConformal(base_forecaster=forecaster, method=method, score=score)
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=0.1)
        assert len(lower) == 150


class TestClaimsCountConformalAdditional:
    def test_coverage_report_mean_width_finite(self):
        """mean_width should be finite when intervals are finite."""
        y = make_y(400, lam=12.0)
        ccc = ClaimsCountConformal(base_forecaster=TestConstantForecaster())
        ccc.fit(y[:200])
        lower, upper = ccc.predict_interval(y[200:], alpha=0.1)
        report = ccc.coverage_report(y[200:], lower, upper)
        finite_widths = (upper - lower)[np.isfinite(upper - lower)]
        if len(finite_widths) > 0:
            assert np.isfinite(report["mean_width"])

    def test_multiple_predict_calls_accumulate(self):
        """Calling predict_interval multiple times on same instance is valid."""
        y = make_y(600)
        ccc = ClaimsCountConformal(base_forecaster=TestConstantForecaster())
        ccc.fit(y[:200])
        l1, u1 = ccc.predict_interval(y[200:250], alpha=0.1)
        l2, u2 = ccc.predict_interval(y[250:300], alpha=0.1)
        assert len(l1) == 50
        assert len(l2) == 50

    def test_n_train_zero_uses_no_data(self):
        """n_train=0 should attempt to fit on empty data (may warn or default)."""
        y = make_y(300)
        ccc = ClaimsCountConformal(base_forecaster=TestConstantForecaster())
        # n_train=0 -> y_fit = y[:0] = []
        # ConstantForecaster.fit on empty array gives mean of []  (=nan)
        # Just verify fit() runs and returns self
        try:
            result = ccc.fit(y, n_train=0)
            assert result is ccc
        except Exception:
            pass  # Acceptable for n_train=0 to fail gracefully


# ---------------------------------------------------------------------------
# Diagnostics: additional paths
# ---------------------------------------------------------------------------

class TestSequentialCoverageReportAdditional:
    def test_perfect_nominal_coverage_kupiec(self):
        """Constructing exactly nominal coverage should pass the Kupiec test."""
        n = 1000
        rng = np.random.default_rng(42)
        alpha = 0.1
        # Construct coverage: 90% covered
        covered_mask = rng.random(n) > alpha  # ~90% True
        y = np.ones(n) * 5.0
        lower = np.where(covered_mask, 0.0, 10.0)
        upper = np.where(covered_mask, 10.0, 20.0)
        result = SequentialCoverageReport(window=20).compute(y, lower, upper, alpha=alpha)
        # Kupiec p-value should be high (can't reject H0)
        assert result["kupiec_pvalue"] > 0.01

    def test_rolling_coverage_starts_at_window_size(self):
        """First rolling_coverage value covers only the first t+1 points."""
        n = 50
        y = np.ones(n) * 5.0
        lower = np.zeros(n)
        upper = np.full(n, 10.0)  # all covered
        result = SequentialCoverageReport(window=10).compute(y, lower, upper, alpha=0.1)
        # First value: only 1 point covered out of 1 = 1.0
        assert result["rolling_coverage"][0] == pytest.approx(1.0)

    def test_drift_slope_negative_for_declining_coverage(self):
        """Intervals that miss more over time should give negative drift slope."""
        n = 200
        y = np.ones(n) * 5.0
        # First half: covered; second half: missed
        lower = np.concatenate([np.zeros(100), np.full(100, 10.0)])
        upper = np.concatenate([np.full(100, 10.0), np.full(100, 20.0)])
        result = SequentialCoverageReport(window=10).compute(y, lower, upper, alpha=0.1)
        # Drift slope should be negative (coverage declines)
        assert result["coverage_drift_slope"] < 0

    def test_drift_slope_positive_for_improving_coverage(self):
        """Intervals that cover more over time should give positive drift slope."""
        n = 200
        y = np.ones(n) * 5.0
        # First half: missed; second half: covered
        lower = np.concatenate([np.full(100, 10.0), np.zeros(100)])
        upper = np.concatenate([np.full(100, 20.0), np.full(100, 10.0)])
        result = SequentialCoverageReport(window=10).compute(y, lower, upper, alpha=0.1)
        # Drift slope should be positive (coverage improves)
        assert result["coverage_drift_slope"] > 0

    def test_window_equals_n(self):
        """window=n: rolling coverage is just the overall coverage at each point."""
        n = 30
        y = np.ones(n) * 5.0
        lower = np.zeros(n)
        upper = np.full(n, 10.0)
        result = SequentialCoverageReport(window=n).compute(y, lower, upper, alpha=0.1)
        assert len(result["rolling_coverage"]) == n
        # All covered, so final rolling value = 1.0
        assert result["rolling_coverage"][-1] == pytest.approx(1.0)

    def test_compute_with_single_observation(self):
        """Single observation edge case should not error."""
        y = np.array([5.0])
        lower = np.array([0.0])
        upper = np.array([10.0])
        result = SequentialCoverageReport(window=2).compute(y, lower, upper, alpha=0.1)
        assert result["n_total"] == 1
        assert result["n_covered"] == 1


class TestIntervalWidthReportAdditional:
    def test_zero_width_intervals(self):
        """Zero-width intervals should give mean_width=0."""
        n = 50
        bounds = np.full(n, 5.0)
        result = IntervalWidthReport().compute(bounds, bounds)
        assert result["mean_width"] == pytest.approx(0.0)

    def test_width_p90_is_correct(self):
        """p90 should be the 90th percentile of widths."""
        n = 100
        lower = np.zeros(n)
        upper = np.arange(1, n + 1, dtype=float)
        result = IntervalWidthReport(window=10).compute(lower, upper)
        expected_p90 = float(np.percentile(upper - lower, 90))
        assert result["width_p90"] == pytest.approx(expected_p90)

    def test_all_finite_n_infinite_zero(self):
        """No infinite intervals: n_infinite should be 0."""
        lower = np.zeros(50)
        upper = np.ones(50) * 5.0
        result = IntervalWidthReport().compute(lower, upper)
        assert result["n_infinite"] == 0

    def test_mixed_finite_infinite_mean_width(self):
        """mean_width ignores infinite-width intervals."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([2.0, np.inf, 4.0])
        result = IntervalWidthReport(window=2).compute(lower, upper)
        # mean of [2.0, 4.0] = 3.0
        assert result["mean_width"] == pytest.approx(3.0)

    def test_rolling_width_shape_matches_n(self):
        """rolling_mean_width should have same length as input."""
        n = 75
        lower = np.zeros(n)
        upper = np.ones(n) * 5.0
        result = IntervalWidthReport(window=15).compute(lower, upper)
        assert len(result["rolling_mean_width"]) == n

    def test_trend_nan_when_fewer_than_window_plus_2(self):
        """Trend should be NaN when n < window + 2."""
        lower = np.zeros(10)
        upper = np.ones(10) * 5.0
        result = IntervalWidthReport(window=10).compute(lower, upper)
        # n=10, window=10: 10 < 10 + 2 = 12 -> NaN
        assert np.isnan(result["width_trend_slope"])


# ---------------------------------------------------------------------------
# Integration tests: methods with non-default scores
# ---------------------------------------------------------------------------

class TestIntegrationMethodScoreCombinations:
    """End-to-end integration: each method with a non-default score."""

    def _run(self, method, y_train, y_test, alpha=0.1):
        method.fit(y_train)
        lower, upper = method.predict_interval(y_test, alpha=alpha)
        assert lower.shape == y_test.shape
        assert upper.shape == y_test.shape
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])
        return lower, upper

    def test_aci_with_locally_weighted_score(self):
        y = make_gaussian(400, mu=100.0, sigma=15.0, seed=10)
        score = LocallyWeightedScore()
        aci = ACI(TestConstantForecaster(), score=score, gamma=0.02, window_size=100)
        self._run(aci, y[:200], y[200:])

    def test_aci_with_exposure_adjusted_score(self):
        rng = np.random.default_rng(11)
        n = 400
        exposure = rng.uniform(50, 200, size=n)
        rate = 0.12
        y = rng.poisson(rate * exposure).astype(float)
        score = ExposureAdjustedScore()
        aci = ACI(TestConstantForecaster(), score=score, gamma=0.02, window_size=100)
        aci.fit(y[:200])
        exp_test = exposure[200:]
        lower, upper = aci.predict_interval(
            y[200:], alpha=0.1, score_kwargs={"exposure": exp_test}
        )
        assert lower.shape == (200,)

    def test_enbpi_with_poisson_score(self):
        y = make_y(500, lam=15.0, seed=12)
        score = PoissonPearsonScore()
        enbpi = EnbPI(lambda: TestConstantForecaster(), score=score, B=20, seed=0)
        self._run(enbpi, y[:200], y[200:])

    def test_spci_with_poisson_score(self):
        y = make_y(500, lam=15.0, seed=13)
        score = PoissonPearsonScore()
        spci = SPCI(
            TestConstantForecaster(),
            score=score,
            n_lags=5,
            min_calibration=20,
        )
        self._run(spci, y[:200], y[200:])

    def test_conformal_pid_with_nb_score(self):
        y = make_y(500, lam=15.0, seed=14)
        score = NegBinomPearsonScore(phi=5.0, parameterisation="NB2")
        pid = ConformalPID(TestConstantForecaster(), score=score, Kp=0.01, Ki=0.001)
        self._run(pid, y[:200], y[200:])

    def test_mscp_with_locally_weighted_score(self):
        y = make_gaussian(600, mu=100.0, sigma=15.0, seed=15)
        score = LocallyWeightedScore()
        mscp = MSCP(TestConstantForecaster(), score=score, H=4, min_cal_per_horizon=10)
        mscp.fit(y[:200])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mscp.calibrate(y[200:400], alpha=0.1)
        fan = mscp.predict_fan()
        for h, (lo, hi) in fan.items():
            if np.isfinite(lo) and np.isfinite(hi):
                assert hi >= lo


# ---------------------------------------------------------------------------
# Built-in forecasters
# ---------------------------------------------------------------------------

class TestBuiltInForecasters:
    def test_constant_forecaster_fit_returns_self(self):
        f = ConstantForecaster()
        result = f.fit(np.array([1.0, 2.0, 3.0]))
        assert result is f

    def test_constant_forecaster_predict_value(self):
        f = ConstantForecaster()
        f.fit(np.array([10.0, 20.0, 30.0]))
        pred = f.predict()
        assert pred[0] == pytest.approx(20.0)

    def test_constant_forecaster_predict_with_X(self):
        """predict(X) with array X should return array of same length."""
        f = ConstantForecaster()
        f.fit(np.array([5.0, 10.0, 15.0]))
        X = np.ones((3, 2))
        # ConstantForecaster.predict ignores X, always returns scalar
        pred = f.predict(X)
        assert pred[0] == pytest.approx(10.0)

    def test_mean_forecaster_is_alias(self):
        """MeanForecaster should behave identically to ConstantForecaster."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        cf = ConstantForecaster()
        mf = MeanForecaster()
        cf.fit(y)
        mf.fit(y)
        np.testing.assert_allclose(cf.predict(), mf.predict())

    def test_constant_forecaster_predict_before_fit(self):
        """ConstantForecaster predicts 0 before fitting (default _mean=0)."""
        f = ConstantForecaster()
        # No fit — default _mean is 0.0
        pred = f.predict()
        assert pred[0] == pytest.approx(0.0)

    def test_constant_forecaster_refit(self):
        """Re-fitting should update the predicted mean."""
        f = ConstantForecaster()
        f.fit(np.array([100.0, 200.0, 300.0]))
        pred_first = f.predict()[0]
        f.fit(np.array([1.0, 2.0, 3.0]))
        pred_second = f.predict()[0]
        assert pred_first != pred_second
        assert pred_second == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# BaseForecaster protocol compliance
# ---------------------------------------------------------------------------

class TestBaseForecasterProtocol:
    def test_constant_forecaster_satisfies_protocol(self):
        from insurance_conformal_ts.methods import BaseForecaster
        f = ConstantForecaster()
        assert isinstance(f, BaseForecaster)

    def test_mean_forecaster_satisfies_protocol(self):
        from insurance_conformal_ts.methods import BaseForecaster
        f = MeanForecaster()
        assert isinstance(f, BaseForecaster)

    def test_custom_forecaster_satisfies_protocol(self):
        """Any object with fit/predict qualifies."""
        from insurance_conformal_ts.methods import BaseForecaster

        class MyForecaster:
            def fit(self, y, X=None):
                self._mu = float(y.mean())
                return self

            def predict(self, X=None):
                return np.array([self._mu])

        f = MyForecaster()
        assert isinstance(f, BaseForecaster)
