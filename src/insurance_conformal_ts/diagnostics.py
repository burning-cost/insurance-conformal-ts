"""
diagnostics.py
==============

Diagnostics for sequential conformal prediction intervals.

Coverage validity is not binary in the sequential setting. You want to
know not just whether nominal coverage is achieved overall, but:

- Whether coverage drifts over time (suggesting the method is slow to adapt).
- Whether there are systematic periods of miscoverage (e.g. high-severity
  years that confound the calibration).
- Whether the intervals are getting wider over time (possible windup in
  ACI or ConformalPID).

These tools are designed for UK pricing team presentations: numbers first,
then plots if needed.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy import stats


class SequentialCoverageReport:
    """Rolling coverage analysis for sequential conformal intervals.

    Computes:

    - Overall empirical coverage.
    - Rolling coverage rate over a sliding window.
    - Coverage drift detection (linear trend in rolling coverage).
    - Kupiec proportion-of-failures (POF) test for correct coverage.

    Parameters
    ----------
    window:
        Rolling window size for coverage computation. Default 20.

    Examples
    --------
    .. code-block:: python

        report = SequentialCoverageReport(window=20)
        result = report.compute(y, lower, upper, alpha=0.1)
        print(result["kupiec_pvalue"])  # > 0.05 => can't reject correct coverage
    """

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError("window must be at least 2.")
        self.window = window

    def compute(
        self,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.1,
    ) -> dict:
        """Compute all coverage metrics.

        Parameters
        ----------
        y:
            Observed values, shape (n,).
        lower, upper:
            Prediction interval bounds, shape (n,).
        alpha:
            Nominal miscoverage level.

        Returns
        -------
        dict
            Keys:

            - ``overall_coverage``: float, empirical coverage rate.
            - ``nominal_coverage``: float, ``1 - alpha``.
            - ``rolling_coverage``: np.ndarray, rolling coverage rates,
              shape (n - window + 1,).
            - ``coverage_drift_slope``: float, OLS slope of rolling
              coverage over time (positive = improving, negative = degrading).
            - ``coverage_drift_pvalue``: float, p-value for slope != 0.
            - ``kupiec_stat``: float, Kupiec POF LR test statistic.
            - ``kupiec_pvalue``: float, p-value for Kupiec test (chi-sq df=1).
              p > 0.05 means cannot reject correct coverage.
            - ``n_covered``: int.
            - ``n_total``: int.
        """
        y = np.asarray(y, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        n = len(y)

        covered = (y >= lower) & (y <= upper)
        n_covered = int(np.sum(covered))
        overall_coverage = n_covered / n

        # Rolling coverage
        rolling = np.array(
            [
                np.mean(covered[max(0, t - self.window + 1) : t + 1])
                for t in range(n)
            ]
        )

        # Coverage drift: OLS on rolling coverage
        drift_slope = np.nan
        drift_pvalue = np.nan
        if n >= self.window + 2:
            t_idx = np.arange(n, dtype=float)
            slope, intercept, r, p, se = stats.linregress(t_idx, rolling)
            drift_slope = float(slope)
            drift_pvalue = float(p)

        # Kupiec POF test
        kupiec_stat, kupiec_pvalue = self._kupiec_test(
            n_covered, n, 1.0 - alpha
        )

        return {
            "overall_coverage": overall_coverage,
            "nominal_coverage": 1.0 - alpha,
            "rolling_coverage": rolling,
            "coverage_drift_slope": drift_slope,
            "coverage_drift_pvalue": drift_pvalue,
            "kupiec_stat": kupiec_stat,
            "kupiec_pvalue": kupiec_pvalue,
            "n_covered": n_covered,
            "n_total": n,
        }

    @staticmethod
    def _kupiec_test(n_covered: int, n: int, p_nominal: float) -> tuple[float, float]:
        """Kupiec proportion-of-failures likelihood ratio test.

        H0: empirical coverage = nominal coverage.
        Test statistic is chi-squared with 1 degree of freedom.

        Parameters
        ----------
        n_covered:
            Number of observations within the interval.
        n:
            Total observations.
        p_nominal:
            Nominal coverage probability.

        Returns
        -------
        stat, pvalue:
            LR test statistic and p-value.
        """
        p_hat = n_covered / n
        # Clip to avoid log(0)
        p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
        p_nom = np.clip(p_nominal, 1e-10, 1 - 1e-10)

        n_miss = n - n_covered
        ll_null = (
            n_covered * np.log(p_nom)
            + n_miss * np.log(1 - p_nom)
        )
        ll_alt = (
            n_covered * np.log(p_hat)
            + n_miss * np.log(1 - p_hat)
        )
        lr_stat = -2 * (ll_null - ll_alt)
        pvalue = float(stats.chi2.sf(lr_stat, df=1))
        return float(lr_stat), pvalue


class IntervalWidthReport:
    """Interval width analysis for sequential prediction intervals.

    Monitors interval efficiency: how wide are the intervals on average,
    and are they widening over time (which would indicate poor adaptation)?

    Parameters
    ----------
    window:
        Rolling window for width statistics. Default 20.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        """Compute interval width statistics.

        Parameters
        ----------
        lower, upper:
            Prediction interval bounds, shape (n,).

        Returns
        -------
        dict
            Keys:

            - ``mean_width``: float.
            - ``median_width``: float.
            - ``width_p90``: float, 90th percentile width.
            - ``rolling_mean_width``: np.ndarray.
            - ``width_trend_slope``: float, OLS slope of rolling width.
            - ``width_trend_pvalue``: float.
            - ``n_infinite``: int, count of infinite-width intervals.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        width = upper - lower

        finite_mask = np.isfinite(width)
        n_infinite = int(np.sum(~finite_mask))
        w_finite = width[finite_mask]

        mean_width = float(np.mean(w_finite)) if len(w_finite) > 0 else np.inf
        median_width = float(np.median(w_finite)) if len(w_finite) > 0 else np.inf
        p90 = float(np.percentile(w_finite, 90)) if len(w_finite) > 0 else np.inf

        n = len(width)
        rolling_w = np.array(
            [
                np.mean(
                    width[max(0, t - self.window + 1) : t + 1][
                        np.isfinite(width[max(0, t - self.window + 1) : t + 1])
                    ]
                )
                for t in range(n)
            ]
        )

        trend_slope = np.nan
        trend_pvalue = np.nan
        if n >= self.window + 2:
            t_idx = np.arange(n, dtype=float)
            finite_roll = np.isfinite(rolling_w)
            if np.sum(finite_roll) >= 3:
                slope, _, _, p, _ = stats.linregress(
                    t_idx[finite_roll], rolling_w[finite_roll]
                )
                trend_slope = float(slope)
                trend_pvalue = float(p)

        return {
            "mean_width": mean_width,
            "median_width": median_width,
            "width_p90": p90,
            "rolling_mean_width": rolling_w,
            "width_trend_slope": trend_slope,
            "width_trend_pvalue": trend_pvalue,
            "n_infinite": n_infinite,
        }


def plot_fan_chart(
    y: np.ndarray | None,
    fan: dict[int, tuple[float, float]],
    y_hat: np.ndarray | None = None,
    origin_index: int = 0,
    title: str = "Multi-step Prediction Fan Chart",
    ax=None,
):
    """Plot a multi-step prediction fan chart.

    Visualises horizon-specific prediction intervals as a fan (shaded
    region narrowing or widening with horizon).

    Parameters
    ----------
    y:
        Historical observations to plot behind the fan. Shape (n,).
        If None, only the fan is plotted.
    fan:
        Output of ``MSCP.predict_fan()``: dict mapping h -> (lower, upper).
    y_hat:
        Point forecast for each horizon. Shape (H,). Optional.
    origin_index:
        Index (time step) at which forecasting starts. Default 0.
    title:
        Plot title.
    ax:
        Matplotlib axes. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes (useful for further customisation).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-conformal-ts[plots]"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    horizons = sorted(fan.keys())
    H = max(horizons)

    lower_arr = np.array([fan[h][0] for h in horizons])
    upper_arr = np.array([fan[h][1] for h in horizons])
    x_fan = np.array([origin_index + h for h in horizons])

    # Historical observations
    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        x_hist = np.arange(len(y_arr))
        ax.plot(x_hist, y_arr, color="steelblue", linewidth=1.5, label="Observed")

    # Fan
    ax.fill_between(
        x_fan,
        lower_arr,
        upper_arr,
        alpha=0.3,
        color="orange",
        label="Prediction interval",
    )
    ax.plot(x_fan, lower_arr, "--", color="orange", linewidth=0.8)
    ax.plot(x_fan, upper_arr, "--", color="orange", linewidth=0.8)

    # Point forecast
    if y_hat is not None:
        y_hat_arr = np.asarray(y_hat, dtype=float)
        ax.plot(x_fan[: len(y_hat_arr)], y_hat_arr, "o-", color="red",
                linewidth=1.5, markersize=4, label="Point forecast")

    # Origin marker
    ax.axvline(x=origin_index, color="grey", linestyle=":", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(loc="best")

    return ax
