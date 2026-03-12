"""
insurance.py
============

Insurance-specific conformal prediction wrappers.

These classes handle the end-to-end workflow that a UK pricing team
actually encounters: fitting a GLM (Poisson for counts, Gamma for
severity), producing sequential prediction intervals via a chosen
conformal method, and reporting coverage.

The wrappers are thin — they handle data preparation and forecaster
wiring, then delegate to the methods in ``methods.py``. The intent is
that teams can swap the underlying conformal method without changing
the rest of their code.

Base Forecasters
----------------
Each wrapper includes a minimal default base forecaster (Poisson GLM,
Gamma GLM) so the library is usable out of the box. For production,
you will typically want to supply your own forecaster — one that
includes features, credibility adjustments, or reinsurance terms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    ExposureAdjustedScore,
    NonConformityScore,
    PoissonPearsonScore,
)
from insurance_conformal_ts.methods import ACI, BaseForecaster


# ---------------------------------------------------------------------------
# Built-in base forecasters
# ---------------------------------------------------------------------------

class _PoissonGLMForecaster:
    """Poisson GLM base forecaster using statsmodels.

    Fits a Poisson GLM with optional log(exposure) offset. The predict
    method returns expected claim counts (mu = E * exp(X @ beta)).

    Parameters
    ----------
    exposure:
        Log-offset per period. If provided, log(exposure) is included
        as an offset in the GLM. Shape (n,).
    """

    def __init__(self, exposure: np.ndarray | None = None) -> None:
        self.exposure = exposure
        self._model = None
        self._result = None

    def fit(
        self, y: np.ndarray, X: np.ndarray | None = None
    ) -> "_PoissonGLMForecaster":
        import statsmodels.api as sm

        y = np.asarray(y, dtype=float)
        if X is None:
            X_design = np.ones((len(y), 1))
        else:
            X_design = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")

        kwargs: dict[str, Any] = {}
        if self.exposure is not None:
            E = np.asarray(self.exposure[: len(y)], dtype=float)
            kwargs["offset"] = np.log(np.maximum(E, 1e-6))

        self._model = sm.GLM(
            y,
            X_design,
            family=sm.families.Poisson(),
            **kwargs,
        )
        self._result = self._model.fit(disp=False)
        self._X_design_cols = X_design.shape[1]
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        import statsmodels.api as sm

        if self._result is None:
            raise RuntimeError("Forecaster not fitted.")

        if X is None:
            X_design = np.ones((1, self._X_design_cols))
        else:
            X_design = sm.add_constant(
                np.atleast_2d(np.asarray(X, dtype=float)),
                has_constant="add",
            )
            if X_design.shape[1] < self._X_design_cols:
                X_design = np.hstack(
                    [X_design, np.ones((len(X_design), self._X_design_cols - X_design.shape[1]))]
                )

        return self._result.predict(X_design)


class _MeanForecaster:
    """Trivial forecaster: predict the training mean.

    Used as a fallback when no base forecaster is provided and
    statsmodels is unavailable. Not suitable for production.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> "_MeanForecaster":
        self._mean = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        n = len(X) if X is not None else 1
        return np.full(n, self._mean)


class _GammaGLMForecaster:
    """Gamma GLM base forecaster with log link.

    Appropriate for modelling average claim severity. Fits log(severity)
    trend with optional features.
    """

    def __init__(self) -> None:
        self._result = None
        self._X_design_cols: int = 1

    def fit(
        self, y: np.ndarray, X: np.ndarray | None = None
    ) -> "_GammaGLMForecaster":
        import statsmodels.api as sm

        y = np.asarray(y, dtype=float)
        if X is None:
            X_design = np.ones((len(y), 1))
        else:
            X_design = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")

        self._model = sm.GLM(
            y,
            X_design,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        )
        self._result = self._model.fit(disp=False)
        self._X_design_cols = X_design.shape[1]
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        import statsmodels.api as sm

        if self._result is None:
            raise RuntimeError("Forecaster not fitted.")
        if X is None:
            X_design = np.ones((1, self._X_design_cols))
        else:
            X_design = sm.add_constant(
                np.atleast_2d(np.asarray(X, dtype=float)),
                has_constant="add",
            )
        return self._result.predict(X_design)


# ---------------------------------------------------------------------------
# Insurance wrappers
# ---------------------------------------------------------------------------

class ClaimsCountConformal:
    """End-to-end conformal intervals for insurance claim counts.

    Handles the full pipeline:

    1. Fit a Poisson GLM (or user-supplied forecaster) on training data.
    2. Apply a sequential conformal method (ACI by default) to produce
       prediction intervals on test/monitoring data.
    3. Report coverage and interval widths.

    The default setup uses a Poisson GLM with log(exposure) offset and
    Poisson Pearson non-conformity scores. This is the right default for
    most UK motor/home claim count series.

    Parameters
    ----------
    base_forecaster:
        Point forecaster implementing ``fit(y, X)`` and ``predict(X)``.
        If None, a Poisson GLM is used.
    method:
        Conformal method instance. If None, uses ``ACI`` with
        ``gamma=0.02``.
    score:
        Non-conformity score. If None, uses ``PoissonPearsonScore``.
    exposure:
        Exposure series for the entire dataset (train + test). Used
        as GLM offset and score denominator. If None, all exposures
        are set to 1.

    Examples
    --------
    .. code-block:: python

        ccc = ClaimsCountConformal(exposure=earned_premiums)
        ccc.fit(y_train, X_train)
        lower, upper = ccc.predict_interval(y_test, X_test, alpha=0.1)
        report = ccc.coverage_report(y_test, lower, upper)
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster | None = None,
        method: Any | None = None,
        score: NonConformityScore | None = None,
        exposure: np.ndarray | None = None,
    ) -> None:
        self.exposure = exposure
        self.score = score if score is not None else PoissonPearsonScore()

        if base_forecaster is None:
            try:
                self._forecaster = _PoissonGLMForecaster(exposure=exposure)
            except ImportError:
                self._forecaster = _MeanForecaster()
        else:
            self._forecaster = base_forecaster

        if method is None:
            self._method: Any = ACI(
                self._forecaster,
                score=self.score,
                gamma=0.02,
                window_size=200,
            )
        else:
            self._method = method

        self._is_fitted = False

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        n_train: int | None = None,
    ) -> "ClaimsCountConformal":
        """Fit the base forecaster.

        Parameters
        ----------
        y:
            Claim count observations, shape (n,).
        X:
            Features. Optional.
        n_train:
            If provided, only the first ``n_train`` observations are
            used for fitting (the remainder form the calibration period).
            If None, all data is used for training.

        Returns
        -------
        ClaimsCountConformal
            Self.
        """
        y = np.asarray(y, dtype=float)
        y_fit = y[:n_train] if n_train is not None else y
        X_fit = X[:n_train] if (X is not None and n_train is not None) else X
        self._method.fit(y_fit, X_fit)
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
        exposure: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce sequential prediction intervals.

        Parameters
        ----------
        y:
            Test claim counts (used to update online calibration).
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.
        exposure:
            Test-period exposures (if different from training exposure).

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")

        score_kwargs: dict = {}
        if exposure is not None:
            score_kwargs["exposure"] = np.asarray(exposure, dtype=float)

        return self._method.predict_interval(y, X, alpha=alpha, score_kwargs=score_kwargs)

    def coverage_report(
        self,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        """Compute basic coverage statistics.

        Parameters
        ----------
        y:
            Observed values.
        lower, upper:
            Prediction interval bounds.

        Returns
        -------
        dict
            Keys: ``coverage``, ``mean_width``, ``n``.
        """
        y = np.asarray(y, dtype=float)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        return {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width[np.isfinite(width)])),
            "n": len(y),
        }


class LossRatioConformal:
    """Sequential conformal intervals for loss ratio time series.

    Loss ratio = incurred losses / earned premium. Loss ratios are
    continuous, typically in [0, 3] for most lines, and can exhibit
    strong seasonality and calendar-year trends.

    Uses absolute residual scores by default (no distributional
    assumption). Supply a ``LocallyWeightedScore`` if you have a
    volatility estimate.

    Parameters
    ----------
    base_forecaster:
        Point forecaster for the loss ratio series.
    method:
        Conformal method instance. If None, uses ``ACI``.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``
        (clip_lower=False — loss ratios can theoretically be zero
        but clipping isn't appropriate here).
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster | None = None,
        method: Any | None = None,
        score: NonConformityScore | None = None,
    ) -> None:
        self.score = score if score is not None else AbsoluteResidualScore(clip_lower=False)

        if base_forecaster is None:
            self._forecaster: BaseForecaster = _MeanForecaster()
        else:
            self._forecaster = base_forecaster

        if method is None:
            self._method: Any = ACI(
                self._forecaster,
                score=self.score,
                gamma=0.02,
                window_size=100,
            )
        else:
            self._method = method

        self._is_fitted = False

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> "LossRatioConformal":
        """Fit the base forecaster on loss ratio training data.

        Parameters
        ----------
        y:
            Loss ratio observations, shape (n_train,).
        X:
            Features. Optional.

        Returns
        -------
        LossRatioConformal
            Self.
        """
        y = np.asarray(y, dtype=float)
        self._method.fit(y, X)
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce sequential prediction intervals for loss ratios.

        Parameters
        ----------
        y:
            Test loss ratio observations.
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")
        return self._method.predict_interval(y, X, alpha=alpha)

    def coverage_report(
        self,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        """Compute coverage statistics.

        Returns
        -------
        dict
            Keys: ``coverage``, ``mean_width``, ``n``.
        """
        y = np.asarray(y, dtype=float)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        return {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width[np.isfinite(width)])),
            "n": len(y),
        }


class SeverityConformal:
    """Sequential conformal intervals for claim severity time series.

    Severity = average claim cost. Typically modelled as Gamma or
    log-normal. This wrapper uses a Gamma GLM by default.

    Claim severities are right-skewed and strictly positive. The
    default absolute residual score is replaced here with a locally
    weighted score if you supply sigma estimates, which gives tighter
    intervals during high-severity periods.

    Parameters
    ----------
    base_forecaster:
        Point forecaster for severity. If None, uses a Gamma GLM.
    method:
        Conformal method. If None, uses ``ACI``.
    score:
        Non-conformity score. Defaults to ``AbsoluteResidualScore``
        with clip_lower=True (severity >= 0).
    """

    def __init__(
        self,
        base_forecaster: BaseForecaster | None = None,
        method: Any | None = None,
        score: NonConformityScore | None = None,
    ) -> None:
        self.score = score if score is not None else AbsoluteResidualScore(clip_lower=True)

        if base_forecaster is None:
            try:
                self._forecaster: BaseForecaster = _GammaGLMForecaster()
            except ImportError:
                self._forecaster = _MeanForecaster()
        else:
            self._forecaster = base_forecaster

        if method is None:
            self._method: Any = ACI(
                self._forecaster,
                score=self.score,
                gamma=0.02,
                window_size=100,
            )
        else:
            self._method = method

        self._is_fitted = False

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> "SeverityConformal":
        """Fit the base forecaster on severity training data.

        Parameters
        ----------
        y:
            Severity observations (average claim cost per period).
        X:
            Features. Optional.

        Returns
        -------
        SeverityConformal
            Self.
        """
        y = np.asarray(y, dtype=float)
        self._method.fit(y, X)
        self._is_fitted = True
        return self

    def predict_interval(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce sequential prediction intervals for severity.

        Parameters
        ----------
        y:
            Test severity observations.
        X:
            Test features. Optional.
        alpha:
            Target miscoverage level.

        Returns
        -------
        lower, upper:
            Prediction interval bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_interval().")
        return self._method.predict_interval(y, X, alpha=alpha)

    def coverage_report(
        self,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict:
        """Compute coverage statistics.

        Returns
        -------
        dict
            Keys: ``coverage``, ``mean_width``, ``n``.
        """
        y = np.asarray(y, dtype=float)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        return {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width[np.isfinite(width)])),
            "n": len(y),
        }
