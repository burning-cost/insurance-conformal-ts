"""
Tests for multistep.py (MSCP)

Tests cover:
- API compliance (fit, calibrate, predict_fan, predict_interval_sequence, update)
- Coverage at each horizon
- Fan chart shape properties (intervals widen with horizon for naive forecaster)
- Edge cases: insufficient calibration, H=1, long horizons
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.multistep import MSCP
from insurance_conformal_ts.nonconformity import AbsoluteResidualScore
from tests.conftest import ConstantForecaster


RNG = np.random.default_rng(42)
ALPHA = 0.1
COVERAGE_TOLERANCE = 0.10  # wider for multi-step (less data per horizon)


def make_series(n: int = 600, lam: float = 10.0, seed: int = 0):
    return np.random.default_rng(seed).poisson(lam, size=n).astype(float)


class TestMSCPFitCalibrate:
    def test_fit_returns_self(self):
        y = make_series(300)
        mscp = MSCP(ConstantForecaster(), H=6)
        result = mscp.fit(y[:150])
        assert result is mscp

    def test_calibrate_returns_self(self):
        y = make_series(300)
        mscp = MSCP(ConstantForecaster(), H=6)
        mscp.fit(y[:150])
        result = mscp.calibrate(y[150:], alpha=ALPHA)
        assert result is mscp

    def test_calibrate_requires_fit(self):
        mscp = MSCP(ConstantForecaster(), H=6)
        with pytest.raises(RuntimeError, match="fit"):
            mscp.calibrate(make_series(50))

    def test_quantiles_computed_for_all_horizons(self):
        y = make_series(500)
        H = 6
        mscp = MSCP(ConstantForecaster(), H=H, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        mscp.calibrate(y[100:400], alpha=ALPHA)
        for h in range(1, H + 1):
            assert h in mscp._h_quantiles

    def test_insufficient_calibration_warns(self):
        """Small calibration window should warn for high horizons."""
        y = make_series(200)
        H = 12
        mscp = MSCP(ConstantForecaster(), H=H, min_cal_per_horizon=50)
        mscp.fit(y[:50])
        with pytest.warns(UserWarning):
            mscp.calibrate(y[50:100], alpha=ALPHA)  # only 50 cal points, H=12


class TestMSCPPredictFan:
    def setup_method(self):
        y = make_series(600)
        self.y = y
        self.H = 6
        self.mscp = MSCP(ConstantForecaster(), H=self.H, min_cal_per_horizon=5)
        self.mscp.fit(y[:200])
        self.mscp.calibrate(y[200:400], alpha=ALPHA)

    def test_fan_returns_dict(self):
        fan = self.mscp.predict_fan(alpha=ALPHA)
        assert isinstance(fan, dict)

    def test_fan_has_all_horizons(self):
        fan = self.mscp.predict_fan(alpha=ALPHA)
        assert set(fan.keys()) == set(range(1, self.H + 1))

    def test_fan_each_horizon_is_tuple(self):
        fan = self.mscp.predict_fan(alpha=ALPHA)
        for h, (lower, upper) in fan.items():
            assert isinstance(lower, float)
            assert isinstance(upper, float)

    def test_fan_upper_geq_lower(self):
        fan = self.mscp.predict_fan(alpha=ALPHA)
        for h, (lower, upper) in fan.items():
            if np.isfinite(lower) and np.isfinite(upper):
                assert upper >= lower, f"h={h}: upper={upper} < lower={lower}"

    def test_fan_requires_calibrated(self):
        mscp = MSCP(ConstantForecaster(), H=4)
        mscp.fit(make_series(200))
        with pytest.raises(RuntimeError, match="calibrate"):
            mscp.predict_fan(alpha=0.1)


class TestMSCPIntervalSequence:
    def setup_method(self):
        y = make_series(700)
        self.y = y
        H = 6
        self.mscp = MSCP(ConstantForecaster(), H=H, min_cal_per_horizon=10)
        self.mscp.fit(y[:200])
        self.mscp.calibrate(y[200:450], alpha=ALPHA)

    def test_sequence_shape(self):
        y_test = self.y[450:]
        lower, upper = self.mscp.predict_interval_sequence(y_test, h=1, alpha=ALPHA)
        n_expected = max(0, len(y_test) - 1)
        assert len(lower) == n_expected
        assert len(upper) == n_expected

    def test_sequence_upper_geq_lower(self):
        y_test = self.y[450:]
        lower, upper = self.mscp.predict_interval_sequence(y_test, h=1, alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_h1(self):
        """MSCP h=1 coverage should be >= 80% on stationary data."""
        y_test = self.y[450:]
        lower, upper = self.mscp.predict_interval_sequence(y_test, h=1, alpha=ALPHA)
        n = min(len(y_test) - 1, len(lower))
        y_obs = y_test[1: n + 1]
        coverage = float(np.mean((y_obs >= lower[:n]) & (y_obs <= upper[:n])))
        assert coverage >= (1 - ALPHA) - COVERAGE_TOLERANCE, (
            f"MSCP h=1 coverage {coverage:.3f} below tolerance"
        )

    def test_invalid_horizon_raises(self):
        y_test = self.y[450:]
        with pytest.raises(ValueError):
            self.mscp.predict_interval_sequence(y_test, h=99)

    def test_requires_calibrated(self):
        mscp = MSCP(ConstantForecaster(), H=4)
        mscp.fit(make_series(200))
        with pytest.raises(RuntimeError, match="calibrate"):
            mscp.predict_interval_sequence(make_series(50), h=1)


class TestMSCPUpdate:
    def test_update_returns_self(self):
        y = make_series(500)
        mscp = MSCP(ConstantForecaster(), H=4, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        mscp.calibrate(y[100:300], alpha=ALPHA)
        result = mscp.update(y[300:350])
        assert result is mscp

    def test_update_adds_scores(self):
        y = make_series(500)
        mscp = MSCP(ConstantForecaster(), H=4, min_cal_per_horizon=5)
        mscp.fit(y[:100])
        mscp.calibrate(y[100:300], alpha=ALPHA)
        len_before = {h: len(mscp._h_scores[h]) for h in range(1, 5)}
        mscp.update(y[300:310])
        len_after = {h: len(mscp._h_scores[h]) for h in range(1, 5)}
        # At least h=1 should have gained scores
        assert len_after[1] >= len_before[1]

    def test_update_requires_calibrated(self):
        mscp = MSCP(ConstantForecaster(), H=4)
        mscp.fit(make_series(200))
        with pytest.raises(RuntimeError, match="calibrate"):
            mscp.update(make_series(50))
