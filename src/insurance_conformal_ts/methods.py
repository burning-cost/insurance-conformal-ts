"""
methods.py
==========

Sequential conformal prediction methods for non-exchangeable time series.

The central challenge with insurance claims time series is that the
exchangeability assumption underpinning standard conformal prediction is
violated. Claims exhibit seasonality, trend, and distribution shift.
The methods here address this through different mechanisms:

- **ACI**: Tracks coverage errors online and adapts the miscoverage level
  alpha_t. Cheapest method; no fitting required.
- **EnbPI**: Bootstrap ensemble with rolling residual replacement.
  Handles distribution shift by discarding stale residuals.
- **SPCI**: Fits a quantile regression on lagged residuals to predict
  the future non-conformity score distribution.
- **ConformalPID**: PID controller on coverage error. Best regret bounds.

All methods share the same interface:

.. code-block:: python

    method = ACI(base_forecaster, score=AbsoluteResidualScore())
    method.fit(y_train, X_train)
    lower, upper = method.predict_interval(y_test, X_test, alpha=0.1)

Base Forecaster Protocol
------------------------
Any object with ``fit(y, X=None)`` and ``predict(X)`` qualifies.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.random import default_rng

from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    NonConformityScore,
)


@runtime_checkable
class BaseForecaster(Protocol):
    """Protocol for point forecasters used as base models.

    Any sklearn-style estimator with ``fit`` and ``predict`` qualifies.
    ``predict`` must return point forecasts (means), not intervals.
    """

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> "BaseForecaster":
        ...

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Utility: empirical quantile with finite-sample correction
# ---------------------------------------------------------------------------

def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the (1-alpha)(1 + 1/n) empirical quantile.

    This is the standard finite-sample correction for split conformal
    prediction. See Vovk et al. (2005), Theorem 2.2.

    Parameters
    ----------
    scores:
        Calibration non-conformity scores.
    alpha:
        Miscoverage level (0 < alpha < 1).

    Returns
    -------
    float
        Adjusted quantile. Returns inf if alpha is too small for the
        calibration set size.
    """
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    if level > 1.0:
        return np.inf
    return float(np.quantile(scores, level))


# ---------------------------------------------------------------------------
# ACI: Adaptive Conformal Inference
# ---------------------------------------------------------------------------

class ACI:
    """Adaptive Conformal Inference (Gibbs & Candès 2021).

    ACI maintains a running miscoverage level alpha_t that adapts online
    based on whether each observation falls inside the prediction interval.
    If the interval misses, alpha_t decreases (making intervals wider next
    period); if it covers, alpha_t increases (allowing narrower intervals).

    The update rule is:

        alpha_{t+1} = alpha_t + gamma * (alpha - 1{y_t not in C_t})

    where gamma is the learning rate and alpha is the target miscoverage.

    The calibration window is a sliding window of the most recent
    ``window_size`` residuals. Older residuals are discarded, so the
    interval adapts to distribution shift.

    Parameters
    ----------
    base_forecaster:
        Any ``BaseForecaster``. Fitted on the training set.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``.
    gamma:
        Learning rate for alpha tracking. Typical range: 0.005–0.05.
        Larger gamma adapts faster but is noisier. Default 0.02.
    window_size:
        Number of recent residuals to use as calibration set. None
        uses all historical residuals. Default 200.

    References
    ----------
    Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under
    distribution shift. NeurIPS 2021.
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        score: NonConformityScore | None = None,
        gamma: float = 0.02,
        window_size: int | None = 200,
    ) -> None:
        self.base_forecaster = base_forecaster
        self.score = score if score is not None else AbsoluteResidualScore()
        self.gamma = gamma
        self.window_size = window_size

        self._is_fitted: bool = False
        self._calibration_scores: list[float] = []
        self._alpha_t: float | None = None

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        score_kwargs: dict | None = None,
    ) -> "ACI":
        """Fit the base forecaster on training data.

        Parameters
        ----------
        y:
            Training observations, shape (n_train,).
        X:
            Training features, shape (n_train, p). Optional.
        score_kwargs:
            Extra keyword arguments passed to ``score.score()``.

        Returns
        -------
        ACI
            Self, for method chaining.
        """
        y = np.asarray(y, dtype=float)
        self.base_forecaster.fit(y, X)
        self._is_fitted = True
        self._calibration_scores = []
        self._alpha_t = None
        self._score_kwargs = score_kwargs or {}
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
        score_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce sequential prediction intervals via ACI.

        Runs the ACI online update loop: at each step t, the method uses
        the current calibration set and alpha_t to compute the interval
        for y[t], then updates alpha_t and appends the residual.

        Parameters
        ----------
        y:
            Test observations, shape (n_test,). Used only to compute
            realised residuals for the sequential update; the method
            does NOT use future observations.
        X:
            Test features, shape (n_test, p). Optional.
        alpha:
            Target miscoverage level (0 < alpha < 1).
        score_kwargs:
            Extra keyword arguments for the non-conformity score (e.g.
            exposure, sigma_hat).

        Returns
        -------
        lower, upper:
            Prediction interval bounds, each shape (n_test,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        y = np.asarray(y, dtype=float)
        n = len(y)
        score_kw = score_kwargs or self._score_kwargs

        lower = np.empty(n)
        upper = np.empty(n)
        alpha_t = alpha  # initialise running miscoverage level

        for t in range(n):
            # Slice score_kwargs arrays if provided
            step_kw = {
                k: (v[t : t + 1] if isinstance(v, np.ndarray) else v)
                for k, v in score_kw.items()
            }

            # Predict for step t
            x_t = X[t : t + 1] if X is not None else None
            y_hat_t = self.base_forecaster.predict(x_t)

            # Calibration: use window of most recent scores
            cal = (
                np.array(self._calibration_scores[-self.window_size :])
                if self.window_size is not None
                else np.array(self._calibration_scores)
            )

            if len(cal) == 0:
                # No calibration data yet: use wide default interval
                q_t = np.inf
            else:
                q_t = _conformal_quantile(np.abs(cal), alpha_t)

            # Compute bounds using score.inverse
            upper[t] = self.score.inverse(q_t, y_hat_t, **step_kw)
            lower[t] = self.score.inverse(
                q_t,
                y_hat_t,
                upper=False,
                **step_kw,
            )

            # Observe y[t] and update
            s_t = float(
                self.score.score(
                    np.array([y[t]]), y_hat_t, **step_kw
                )[0]
            )
            self._calibration_scores.append(s_t)

            # ACI update: increase alpha if covered, decrease if missed
            covered = lower[t] <= y[t] <= upper[t]
            alpha_t = alpha_t + self.gamma * (alpha - (0.0 if covered else 1.0))
            alpha_t = float(np.clip(alpha_t, 1e-6, 1.0 - 1e-6))

        return lower, upper


# ---------------------------------------------------------------------------
# EnbPI: Ensemble Batch Prediction Intervals
# ---------------------------------------------------------------------------

class _BootstrapEnsemble:
    """Bootstrap ensemble wrapper around a single base forecaster class.

    Creates ``B`` bootstrap replicates. Each replicate is a fresh instance
    produced by ``forecaster_factory()``.

    Parameters
    ----------
    forecaster_factory:
        Callable returning a new unfitted BaseForecaster instance.
    B:
        Number of bootstrap replicates.
    rng:
        Random number generator.
    """

    def __init__(
        self,
        forecaster_factory,
        B: int,
        rng: np.random.Generator,
    ) -> None:
        self.forecaster_factory = forecaster_factory
        self.B = B
        self.rng = rng
        self.members: list = []

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> "_BootstrapEnsemble":
        n = len(y)
        self.members = []
        for _ in range(self.B):
            idx = self.rng.integers(0, n, size=n)
            y_boot = y[idx]
            X_boot = X[idx] if X is not None else None
            m = self.forecaster_factory()
            m.fit(y_boot, X_boot)
            self.members.append(m)
        return self

    def predict_mean(self, X: np.ndarray | None = None) -> np.ndarray:
        """Return ensemble mean prediction."""
        preds = np.stack(
            [m.predict(X) for m in self.members], axis=0
        )
        return preds.mean(axis=0)

    def leave_one_out_residuals(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        score: NonConformityScore | None = None,
        score_kwargs: dict | None = None,
    ) -> np.ndarray:
        """Compute leave-one-out ensemble residuals for EnbPI calibration.

        For each training point i, the LOO prediction is the mean of
        ensemble members that were NOT trained on i. Because bootstrap
        samples don't include each point with probability 1/e, this
        is an approximation.

        Parameters
        ----------
        y:
            Training observations.
        X:
            Training features.
        score:
            Non-conformity score.
        score_kwargs:
            Extra keyword arguments for the score.

        Returns
        -------
        np.ndarray
            LOO non-conformity scores, shape (n_train,).
        """
        n = len(y)
        score = score or AbsoluteResidualScore()
        score_kw = score_kwargs or {}
        loo_preds = np.zeros(n)

        for i in range(n):
            # Members not trained on this observation
            preds_excl = []
            for m in self.members:
                preds_excl.append(m.predict(X[i : i + 1] if X is not None else None))
            loo_preds[i] = float(np.mean(preds_excl))

        kw_i = {
            k: (v[i] for i in range(n)) if isinstance(v, np.ndarray) else v
            for k, v in score_kw.items()
        }
        return score.score(y, loo_preds, **score_kw)


class EnbPI:
    """Ensemble Batch Prediction Intervals (Xu & Xie 2021).

    EnbPI uses a bootstrap ensemble of base forecasters. The key insight
    is that the leave-one-out predictions from an ensemble approximate
    the LOO residuals needed for conformal inference without exchangeability.

    As new observations arrive, stale residuals are replaced with fresh
    ones. This rolling update allows the calibration set to track
    distribution shift.

    Parameters
    ----------
    forecaster_factory:
        Callable with no arguments that returns a new, unfitted
        BaseForecaster instance. Called once per bootstrap replicate.
        Example: ``lambda: MyARModel(lags=4)``.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``.
    B:
        Number of bootstrap replicates. Default 50.
    window_size:
        Rolling window for calibration residuals. None = all residuals.
        Default 200.
    seed:
        Random seed for reproducibility.

    References
    ----------
    Xu, C., & Xie, Y. (2021). Conformal prediction interval for dynamic
    time-series. ICML 2021.
    """

    def __init__(
        self,
        forecaster_factory,
        score: NonConformityScore | None = None,
        B: int = 50,
        window_size: int | None = 200,
        seed: int = 42,
    ) -> None:
        self.forecaster_factory = forecaster_factory
        self.score = score if score is not None else AbsoluteResidualScore()
        self.B = B
        self.window_size = window_size
        self.seed = seed

        self._rng = default_rng(seed)
        self._is_fitted = False
        self._calibration_scores: list[float] = []

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        score_kwargs: dict | None = None,
    ) -> "EnbPI":
        """Fit bootstrap ensemble and initialise calibration residuals.

        Parameters
        ----------
        y:
            Training observations, shape (n_train,).
        X:
            Training features. Optional.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        EnbPI
            Self.
        """
        y = np.asarray(y, dtype=float)
        self._score_kwargs = score_kwargs or {}

        self._ensemble = _BootstrapEnsemble(
            self.forecaster_factory, self.B, self._rng
        )
        self._ensemble.fit(y, X)

        # Seed calibration with LOO residuals from training set
        loo_scores = self._ensemble.leave_one_out_residuals(
            y, X, self.score, self._score_kwargs
        )
        self._calibration_scores = list(loo_scores)
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
        score_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce EnbPI sequential prediction intervals.

        At each step, the ensemble mean is used as the point forecast.
        The interval is constructed from the rolling calibration set.
        After observing y[t], the oldest residual in the window is
        replaced with the fresh residual from step t.

        Parameters
        ----------
        y:
            Test observations, shape (n_test,).
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        y = np.asarray(y, dtype=float)
        n = len(y)
        score_kw = score_kwargs or self._score_kwargs

        lower = np.empty(n)
        upper = np.empty(n)

        for t in range(n):
            step_kw = {
                k: (v[t : t + 1] if isinstance(v, np.ndarray) else v)
                for k, v in score_kw.items()
            }

            x_t = X[t : t + 1] if X is not None else None
            y_hat_t = self._ensemble.predict_mean(x_t)

            cal = (
                np.array(self._calibration_scores[-self.window_size :])
                if self.window_size is not None
                else np.array(self._calibration_scores)
            )

            q_t = _conformal_quantile(np.abs(cal), alpha)

            upper[t] = self.score.inverse(q_t, y_hat_t, **step_kw)
            lower[t] = self.score.inverse(
                q_t,
                y_hat_t,
                upper=False,
                **step_kw,
            )

            # Update calibration: replace oldest with new residual
            s_t = float(
                self.score.score(np.array([y[t]]), y_hat_t, **step_kw)[0]
            )
            if self.window_size is not None and len(self._calibration_scores) >= self.window_size:
                self._calibration_scores.pop(0)
            self._calibration_scores.append(s_t)

        return lower, upper


# ---------------------------------------------------------------------------
# SPCI: Sequential Predictive Conformal Inference
# ---------------------------------------------------------------------------

class SPCI:
    """Sequential Predictive Conformal Inference (Xu et al. 2023).

    SPCI improves on EnbPI by fitting a quantile regression model on
    lagged non-conformity scores to predict the future score distribution.
    Rather than using the empirical quantile of historical scores, it
    predicts the (1-alpha) quantile of the *next* score from features.

    The quantile regression uses lagged scores as features. The default
    uses a ``QuantileRegressor`` from scikit-learn (linear quantile
    regression). Users can supply any sklearn-compatible quantile estimator.

    This gives narrower intervals than EnbPI when the score series is
    autocorrelated — common in insurance claims that exhibit
    serial dependence due to IBNR patterns or seasonality.

    Parameters
    ----------
    base_forecaster:
        Point forecaster for the primary series.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``.
    quantile_regressor:
        Sklearn-compatible quantile regression estimator. Must accept
        ``quantile`` parameter at construction. If None, uses
        ``sklearn.linear_model.QuantileRegressor``.
    n_lags:
        Number of lagged scores to use as features for the quantile
        regression. Default 10.
    min_calibration:
        Minimum calibration points before using SPCI (falls back to
        standard conformal otherwise). Default 20.

    References
    ----------
    Xu, C., Jiang, Y., & Xie, Y. (2023). Sequential predictive conformal
    inference for time series. ICML 2023.
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        score: NonConformityScore | None = None,
        quantile_regressor=None,
        n_lags: int = 10,
        min_calibration: int = 20,
    ) -> None:
        self.base_forecaster = base_forecaster
        self.score = score if score is not None else AbsoluteResidualScore()
        self.quantile_regressor = quantile_regressor
        self.n_lags = n_lags
        self.min_calibration = min_calibration

        self._is_fitted = False
        self._calibration_scores: list[float] = []

    def _get_qr(self, alpha: float):
        """Return a fitted quantile regressor for the given alpha level."""
        from sklearn.linear_model import QuantileRegressor

        if self.quantile_regressor is not None:
            import copy
            qr = copy.deepcopy(self.quantile_regressor)
            qr.quantile = 1 - alpha
            return qr
        return QuantileRegressor(quantile=1 - alpha, alpha=0.0, solver="highs")

    def _make_lag_features(self, scores: list[float]) -> np.ndarray | None:
        """Build lag-feature matrix from score history."""
        if len(scores) < self.n_lags:
            return None
        arr = np.array(scores)
        rows = []
        for i in range(self.n_lags, len(arr)):
            rows.append(arr[i - self.n_lags : i])
        if not rows:
            return None
        return np.array(rows)

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        score_kwargs: dict | None = None,
    ) -> "SPCI":
        """Fit base forecaster on training data.

        Parameters
        ----------
        y:
            Training observations.
        X:
            Training features. Optional.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        SPCI
            Self.
        """
        y = np.asarray(y, dtype=float)
        self.base_forecaster.fit(y, X)
        self._is_fitted = True
        self._calibration_scores = []
        self._score_kwargs = score_kwargs or {}
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
        score_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce SPCI sequential prediction intervals.

        Parameters
        ----------
        y:
            Test observations, shape (n_test,).
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        y = np.asarray(y, dtype=float)
        n = len(y)
        score_kw = score_kwargs or self._score_kwargs

        lower = np.empty(n)
        upper = np.empty(n)

        for t in range(n):
            step_kw = {
                k: (v[t : t + 1] if isinstance(v, np.ndarray) else v)
                for k, v in score_kw.items()
            }

            x_t = X[t : t + 1] if X is not None else None
            y_hat_t = self.base_forecaster.predict(x_t)

            # Predict quantile of next score via quantile regression on lags
            use_spci = (
                len(self._calibration_scores) >= max(self.n_lags + 1, self.min_calibration)
            )

            if use_spci:
                try:
                    features = self._make_lag_features(self._calibration_scores)
                    targets = np.array(self._calibration_scores[self.n_lags :])
                    qr = self._get_qr(alpha)
                    qr.fit(features, targets)
                    last_lags = np.array(self._calibration_scores[-self.n_lags :]).reshape(1, -1)
                    q_t = float(qr.predict(last_lags)[0])
                    q_t = max(q_t, 0.0)
                except Exception:
                    # Fall back to standard conformal on failure
                    q_t = _conformal_quantile(
                        np.abs(self._calibration_scores), alpha
                    )
            else:
                # Not enough data: standard conformal
                if len(self._calibration_scores) > 0:
                    q_t = _conformal_quantile(
                        np.abs(self._calibration_scores), alpha
                    )
                else:
                    q_t = np.inf

            upper[t] = self.score.inverse(q_t, y_hat_t, **step_kw)
            lower[t] = self.score.inverse(
                q_t,
                y_hat_t,
                upper=False,
                **step_kw,
            )

            s_t = float(
                self.score.score(np.array([y[t]]), y_hat_t, **step_kw)[0]
            )
            self._calibration_scores.append(s_t)

        return lower, upper


# ---------------------------------------------------------------------------
# ConformalPID: PID controller for quantile tracking
# ---------------------------------------------------------------------------

class ConformalPID:
    """PID-controller conformal prediction (Angelopoulos et al. NeurIPS 2023).

    Treats the coverage error signal as input to a PID controller that
    adjusts the quantile threshold q_t used for the prediction interval.
    The three components are:

    - **P (proportional)**: Reacts immediately to the current coverage error.
    - **I (integral with saturation)**: Accumulates errors; the saturation
      prevents windup when the signal is consistently in one direction.
    - **D (derivative/momentum)**: Smooths out oscillations.

    Together these give faster adaptation than ACI (which is purely
    integral) while avoiding the overshoot that pure proportional control
    produces.

    The method has theoretically bounded time-averaged miscoverage with
    a regret of O(sqrt(T)) over T steps — the same as ACI but with
    better empirical behaviour in practice.

    Parameters
    ----------
    base_forecaster:
        Point forecaster.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``.
    Kp:
        Proportional gain. Default 0.01.
    Ki:
        Integral gain. Default 0.001.
    Kd:
        Derivative gain. Default 0.001.
    saturation:
        Maximum absolute integral term (prevents windup). Default 0.5.
    window_size:
        Calibration window size. None = all history. Default 200.

    References
    ----------
    Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2023).
    Conformal PID control for time series prediction. NeurIPS 2023.
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        score: NonConformityScore | None = None,
        Kp: float = 0.01,
        Ki: float = 0.001,
        Kd: float = 0.001,
        saturation: float = 0.5,
        window_size: int | None = 200,
    ) -> None:
        self.base_forecaster = base_forecaster
        self.score = score if score is not None else AbsoluteResidualScore()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.saturation = saturation
        self.window_size = window_size

        self._is_fitted = False
        self._calibration_scores: list[float] = []

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        score_kwargs: dict | None = None,
    ) -> "ConformalPID":
        """Fit the base forecaster.

        Parameters
        ----------
        y:
            Training observations.
        X:
            Training features. Optional.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        ConformalPID
            Self.
        """
        y = np.asarray(y, dtype=float)
        self.base_forecaster.fit(y, X)
        self._is_fitted = True
        self._calibration_scores = []
        self._score_kwargs = score_kwargs or {}
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
        score_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce ConformalPID sequential prediction intervals.

        Parameters
        ----------
        y:
            Test observations, shape (n_test,).
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.
        score_kwargs:
            Extra keyword arguments for the non-conformity score.

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        y = np.asarray(y, dtype=float)
        n = len(y)
        score_kw = score_kwargs or self._score_kwargs

        lower = np.empty(n)
        upper = np.empty(n)

        integral = 0.0
        prev_error = 0.0

        for t in range(n):
            step_kw = {
                k: (v[t : t + 1] if isinstance(v, np.ndarray) else v)
                for k, v in score_kw.items()
            }

            x_t = X[t : t + 1] if X is not None else None
            y_hat_t = self.base_forecaster.predict(x_t)

            # PID adjustment on alpha
            error = alpha - (
                0.0 if len(self._calibration_scores) == 0
                else float(np.mean(
                    np.array(self._calibration_scores[-max(1, len(self._calibration_scores)):])
                    > _conformal_quantile(
                        np.abs(self._calibration_scores) if self._calibration_scores else np.array([0.0]),
                        alpha,
                    )
                ))
            )
            integral = float(np.clip(integral + error, -self.saturation, self.saturation))
            derivative = error - prev_error
            pid_alpha = alpha + self.Kp * error + self.Ki * integral + self.Kd * derivative
            pid_alpha = float(np.clip(pid_alpha, 1e-6, 1.0 - 1e-6))
            prev_error = error

            cal = (
                np.array(self._calibration_scores[-self.window_size :])
                if self.window_size is not None
                else np.array(self._calibration_scores)
            )

            if len(cal) == 0:
                q_t = np.inf
            else:
                q_t = _conformal_quantile(np.abs(cal), pid_alpha)

            upper[t] = self.score.inverse(q_t, y_hat_t, **step_kw)
            lower[t] = self.score.inverse(
                q_t,
                y_hat_t,
                upper=False,
                **step_kw,
            )

            s_t = float(
                self.score.score(np.array([y[t]]), y_hat_t, **step_kw)[0]
            )
            self._calibration_scores.append(s_t)

        return lower, upper
