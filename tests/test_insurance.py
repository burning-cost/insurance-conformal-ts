"""
Tests for insurance.py wrappers.

These tests use synthetic data with known DGPs and verify:
- API: fit/predict_interval/coverage_report all work
- Coverage is achieved at nominal level on synthetic stationary data
- Wrappers accept custom base forecasters and methods
- LossRatio and Severity wrappers function independently
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.insurance import (
    ClaimsCountConformal,
    LossRatioConformal,
    SeverityConformal,
)
from insurance_conformal_ts.methods import ACI
from insurance_conformal_ts.nonconformity import AbsoluteResidualScore
from tests.conftest import ConstantForecaster, PoissonMeanForecaster


RNG = np.random.default_rng(42)
ALPHA = 0.1
COVERAGE_TOLERANCE = 0.10


def make_count_series(n: int = 500, lam: float = 12.0, seed: int = 0):
    return np.random.default_rng(seed).poisson(lam, size=n).astype(float)


def make_loss_ratio_series(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    return 0.65 + rng.normal(0, 0.06, size=n)


def make_severity_series(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.gamma(4, 500, size=n)  # mean=2000, shape=4


class TestClaimsCountConformal:
    def test_fit_returns_self(self):
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        result = ccc.fit(y[:200])
        assert result is ccc

    def test_predict_interval_shape(self):
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=PoissonMeanForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_lower_nonnegative(self):
        """Claim counts cannot be negative; lower bound should be >= 0."""
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        assert np.all(lower[np.isfinite(lower)] >= 0)

    def test_coverage_report_keys(self):
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        report = ccc.coverage_report(y[150:], lower, upper)
        assert "coverage" in report
        assert "mean_width" in report
        assert "n" in report

    def test_coverage_report_n(self):
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        report = ccc.coverage_report(y[150:], lower, upper)
        assert report["n"] == 150

    def test_coverage_report_coverage_range(self):
        """Coverage should be in [0, 1]."""
        y = make_count_series(300)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:150])
        lower, upper = ccc.predict_interval(y[150:], alpha=ALPHA)
        report = ccc.coverage_report(y[150:], lower, upper)
        assert 0.0 <= report["coverage"] <= 1.0

    def test_achieves_nominal_coverage(self):
        """ClaimsCountConformal should achieve >= 80% coverage (nominal 90%)."""
        n_train = 200
        n_test = 500
        y = make_count_series(n_train + n_test, lam=12, seed=7)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:n_train])
        lower, upper = ccc.predict_interval(y[n_train:], alpha=ALPHA)
        report = ccc.coverage_report(y[n_train:], lower, upper)
        assert report["coverage"] >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_raises_before_fit(self):
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            ccc.predict_interval(make_count_series(50), alpha=0.1)

    def test_n_train_splits_data(self):
        """n_train parameter should only use first n_train obs for fitting."""
        y = make_count_series(400)
        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y, n_train=200)
        assert ccc._is_fitted

    def test_custom_method(self):
        """Custom ACI method should be accepted."""
        y = make_count_series(400)
        forecaster = ConstantForecaster()
        score = AbsoluteResidualScore()
        method = ACI(forecaster, score=score, gamma=0.03, window_size=100)
        ccc = ClaimsCountConformal(
            base_forecaster=forecaster,
            method=method,
            score=score,
        )
        ccc.fit(y[:200])
        lower, upper = ccc.predict_interval(y[200:], alpha=0.1)
        assert len(lower) == 200

    def test_with_exposure(self):
        """Exposure array should be accepted and not cause errors."""
        rng = np.random.default_rng(5)
        n = 400
        exposure = rng.uniform(50, 200, size=n)
        rate = 0.12
        y = rng.poisson(rate * exposure).astype(float)

        ccc = ClaimsCountConformal(base_forecaster=ConstantForecaster())
        ccc.fit(y[:200])
        lower, upper = ccc.predict_interval(
            y[200:], alpha=ALPHA, exposure=exposure[200:]
        )
        assert len(lower) == 200


class TestLossRatioConformal:
    def test_fit_returns_self(self):
        y = make_loss_ratio_series(300)
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        result = lrc.fit(y[:150])
        assert result is lrc

    def test_predict_interval_shape(self):
        y = make_loss_ratio_series(300)
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        lrc.fit(y[:150])
        lower, upper = lrc.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_loss_ratio_series(300)
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        lrc.fit(y[:150])
        lower, upper = lrc.predict_interval(y[150:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_coverage_achieved(self):
        n_train = 200
        n_test = 400
        y = make_loss_ratio_series(n_train + n_test, seed=9)
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        lrc.fit(y[:n_train])
        lower, upper = lrc.predict_interval(y[n_train:], alpha=ALPHA)
        report = lrc.coverage_report(y[n_train:], lower, upper)
        assert report["coverage"] >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_raises_before_fit(self):
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            lrc.predict_interval(make_loss_ratio_series(50), alpha=0.1)

    def test_coverage_report_returns_dict(self):
        y = make_loss_ratio_series(300)
        lrc = LossRatioConformal(base_forecaster=ConstantForecaster())
        lrc.fit(y[:150])
        lower, upper = lrc.predict_interval(y[150:], alpha=ALPHA)
        report = lrc.coverage_report(y[150:], lower, upper)
        assert isinstance(report, dict)
        assert set(report.keys()) == {"coverage", "mean_width", "n"}


class TestSeverityConformal:
    def test_fit_returns_self(self):
        y = make_severity_series(300)
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        result = sc.fit(y[:150])
        assert result is sc

    def test_predict_interval_shape(self):
        y = make_severity_series(300)
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=ALPHA)
        assert lower.shape == (150,)
        assert upper.shape == (150,)

    def test_upper_geq_lower(self):
        y = make_severity_series(300)
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=ALPHA)
        finite = np.isfinite(lower) & np.isfinite(upper)
        assert np.all(upper[finite] >= lower[finite])

    def test_lower_nonnegative(self):
        """Severity is non-negative; lower bound should be >= 0."""
        y = make_severity_series(300)
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        sc.fit(y[:150])
        lower, upper = sc.predict_interval(y[150:], alpha=ALPHA)
        assert np.all(lower[np.isfinite(lower)] >= 0)

    def test_coverage_achieved(self):
        n_train = 200
        n_test = 400
        y = make_severity_series(n_train + n_test, seed=11)
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        sc.fit(y[:n_train])
        lower, upper = sc.predict_interval(y[n_train:], alpha=ALPHA)
        report = sc.coverage_report(y[n_train:], lower, upper)
        assert report["coverage"] >= (1 - ALPHA) - COVERAGE_TOLERANCE

    def test_raises_before_fit(self):
        sc = SeverityConformal(base_forecaster=ConstantForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            sc.predict_interval(make_severity_series(50), alpha=0.1)
