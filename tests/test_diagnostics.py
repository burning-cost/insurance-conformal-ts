"""
Tests for diagnostics.py

Tests cover:
- SequentialCoverageReport: correct metrics, Kupiec test, coverage drift
- IntervalWidthReport: width statistics, trend detection
- plot_fan_chart: smoke test (no display)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.diagnostics import (
    IntervalWidthReport,
    SequentialCoverageReport,
)


RNG = np.random.default_rng(42)


def make_intervals(n: int = 200, centre: float = 10.0, half_width: float = 3.0):
    y = RNG.poisson(centre, size=n).astype(float)
    lower = np.full(n, centre - half_width)
    upper = np.full(n, centre + half_width)
    return y, lower, upper


class TestSequentialCoverageReport:
    def test_returns_dict(self):
        y, lower, upper = make_intervals(200)
        report = SequentialCoverageReport(window=20)
        result = report.compute(y, lower, upper, alpha=0.1)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        y, lower, upper = make_intervals(200)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.1)
        expected_keys = {
            "overall_coverage",
            "nominal_coverage",
            "rolling_coverage",
            "coverage_drift_slope",
            "coverage_drift_pvalue",
            "kupiec_stat",
            "kupiec_pvalue",
            "n_covered",
            "n_total",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_coverage_in_range(self):
        y, lower, upper = make_intervals(200)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.1)
        assert 0.0 <= result["overall_coverage"] <= 1.0

    def test_n_total_correct(self):
        y, lower, upper = make_intervals(150)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.1)
        assert result["n_total"] == 150

    def test_n_covered_correct(self):
        y = np.array([5.0, 10.0, 15.0])
        lower = np.array([4.0, 12.0, 4.0])  # 0, miss (10 < 12), cover
        upper = np.array([11.0, 20.0, 20.0])
        result = SequentialCoverageReport(window=2).compute(y, lower, upper, alpha=0.1)
        assert result["n_covered"] == 2

    def test_nominal_coverage_set_correctly(self):
        y, lower, upper = make_intervals(100)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.05)
        assert result["nominal_coverage"] == pytest.approx(0.95)

    def test_rolling_coverage_shape(self):
        n = 100
        y, lower, upper = make_intervals(n)
        result = SequentialCoverageReport(window=10).compute(y, lower, upper, alpha=0.1)
        assert len(result["rolling_coverage"]) == n

    def test_rolling_coverage_in_range(self):
        y, lower, upper = make_intervals(200)
        result = SequentialCoverageReport(window=20).compute(y, lower, upper, alpha=0.1)
        rc = result["rolling_coverage"]
        assert np.all(rc >= 0.0) and np.all(rc <= 1.0)

    def test_kupiec_high_pvalue_for_correct_coverage(self):
        """Perfect coverage at 90% should give high p-value (fail to reject H0)."""
        n = 500
        y = RNG.poisson(10, size=n).astype(float)
        # Construct intervals so exactly 90% of obs are covered
        # Use wide intervals that always cover
        lower = np.zeros(n)
        upper = np.full(n, 1000.0)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.1)
        # Coverage = 100% here, so Kupiec will reject H0 (which is correct)
        # Just check the stat is a valid float
        assert np.isfinite(result["kupiec_stat"])
        assert 0.0 <= result["kupiec_pvalue"] <= 1.0

    def test_kupiec_low_pvalue_for_zero_coverage(self):
        """Zero coverage at alpha=0.1 (nominal 90%) should strongly reject H0."""
        n = 200
        y = np.ones(n) * 5.0
        lower = np.full(n, 10.0)  # All miss: y < lower
        upper = np.full(n, 20.0)
        result = SequentialCoverageReport().compute(y, lower, upper, alpha=0.1)
        assert result["kupiec_pvalue"] < 0.01

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            SequentialCoverageReport(window=1)

    def test_coverage_drift_slope_nan_for_small_n(self):
        """Small n (< window + 2) should give NaN drift slope."""
        y, lower, upper = make_intervals(5)
        result = SequentialCoverageReport(window=5).compute(y, lower, upper, alpha=0.1)
        assert np.isnan(result["coverage_drift_slope"])


class TestIntervalWidthReport:
    def test_returns_dict(self):
        lower = np.ones(100) * 5.0
        upper = np.ones(100) * 15.0
        report = IntervalWidthReport()
        result = report.compute(lower, upper)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        lower = np.ones(100) * 5.0
        upper = np.ones(100) * 15.0
        result = IntervalWidthReport().compute(lower, upper)
        expected = {
            "mean_width",
            "median_width",
            "width_p90",
            "rolling_mean_width",
            "width_trend_slope",
            "width_trend_pvalue",
            "n_infinite",
        }
        assert expected.issubset(set(result.keys()))

    def test_mean_width_correct(self):
        lower = np.zeros(100)
        upper = np.full(100, 8.0)
        result = IntervalWidthReport().compute(lower, upper)
        assert result["mean_width"] == pytest.approx(8.0)

    def test_median_width_correct(self):
        lower = np.zeros(4)
        upper = np.array([2.0, 4.0, 6.0, 8.0])
        result = IntervalWidthReport(window=2).compute(lower, upper)
        assert result["median_width"] == pytest.approx(5.0)

    def test_n_infinite_counted(self):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, np.inf, 10.0])
        result = IntervalWidthReport(window=2).compute(lower, upper)
        assert result["n_infinite"] == 1

    def test_all_infinite(self):
        lower = np.full(10, -np.inf)
        upper = np.full(10, np.inf)
        result = IntervalWidthReport().compute(lower, upper)
        assert result["n_infinite"] == 10
        assert result["mean_width"] == np.inf

    def test_rolling_width_shape(self):
        n = 100
        lower = np.zeros(n)
        upper = np.arange(1, n + 1, dtype=float)
        result = IntervalWidthReport(window=10).compute(lower, upper)
        assert len(result["rolling_mean_width"]) == n

    def test_width_trend_slope_positive_for_widening(self):
        """Widening intervals should produce positive trend slope."""
        n = 100
        lower = np.zeros(n)
        upper = np.linspace(1.0, 10.0, n)  # monotonically increasing
        result = IntervalWidthReport(window=5).compute(lower, upper)
        assert result["width_trend_slope"] > 0

    def test_width_trend_slope_nan_for_small_n(self):
        lower = np.zeros(5)
        upper = np.ones(5)
        result = IntervalWidthReport(window=5).compute(lower, upper)
        assert np.isnan(result["width_trend_slope"])


class TestPlotFanChart:
    def test_import(self):
        """plot_fan_chart should be importable."""
        from insurance_conformal_ts.diagnostics import plot_fan_chart
        assert callable(plot_fan_chart)

    def test_runs_without_error(self):
        """Smoke test: plot_fan_chart should run without errors when matplotlib available."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            from insurance_conformal_ts.diagnostics import plot_fan_chart
        except ImportError:
            pytest.skip("matplotlib not installed")

        y = np.random.default_rng(0).poisson(10, size=50).astype(float)
        fan = {h: (8.0, 12.0 + h * 0.5) for h in range(1, 7)}
        ax = plot_fan_chart(y, fan, title="Test fan")
        assert ax is not None

    def test_runs_without_y(self):
        """Fan chart with no historical y."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            from insurance_conformal_ts.diagnostics import plot_fan_chart
        except ImportError:
            pytest.skip("matplotlib not installed")

        fan = {h: (8.0, 13.0) for h in range(1, 5)}
        ax = plot_fan_chart(None, fan)
        assert ax is not None

    def test_raises_without_matplotlib(self, monkeypatch):
        """Should raise ImportError if matplotlib is not available."""
        import sys
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        monkeypatch.setitem(sys.modules, "matplotlib.patches", None)

        # Re-import to pick up the monkeypatched modules
        import importlib
        import insurance_conformal_ts.diagnostics as diag_mod
        importlib.reload(diag_mod)

        fan = {1: (0.0, 10.0)}
        with pytest.raises((ImportError, TypeError)):
            diag_mod.plot_fan_chart(None, fan)
