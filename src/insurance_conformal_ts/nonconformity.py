"""
nonconformity.py
================

Non-conformity scores for insurance count and severity time series.

A non-conformity score maps a (prediction, observation) pair to a scalar
indicating how surprising the observation is given the prediction. Larger
values mean more surprising. The score choice matters for efficiency:
a well-calibrated score concentrates mass near zero, producing narrower
prediction intervals for the same nominal coverage.

Design
------
All scores implement the ``NonConformityScore`` Protocol. This lets every
method in ``methods.py`` accept any score without inheritance overhead.
The protocol requires two methods:

- ``score(y, y_hat, **kwargs) -> np.ndarray``
- ``inverse(s, y_hat, **kwargs) -> np.ndarray``

``inverse`` maps a quantile of the score distribution back to a prediction
interval bound. For symmetric scores this is straightforward; for Pearson
residuals it requires solving for y given the score.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NonConformityScore(Protocol):
    """Protocol for non-conformity score functions.

    Any object implementing ``score`` and ``inverse`` is a valid
    ``NonConformityScore``. No inheritance required.
    """

    def score(self, y: np.ndarray, y_hat: np.ndarray, **kwargs) -> np.ndarray:
        """Compute non-conformity scores.

        Parameters
        ----------
        y:
            Observed values, shape (n,).
        y_hat:
            Point forecasts, shape (n,).
        **kwargs:
            Score-specific extras (e.g. exposure, sigma_hat).

        Returns
        -------
        np.ndarray
            Non-conformity scores, shape (n,). Always non-negative for
            scores used with one-sided quantiles; may be signed for
            two-sided variants.
        """
        ...

    def inverse(self, s: np.ndarray, y_hat: np.ndarray, **kwargs) -> np.ndarray:
        """Map score quantile back to an observation-space bound.

        Parameters
        ----------
        s:
            Score quantile(s), shape (n,) or scalar.
        y_hat:
            Point forecasts at the same steps, shape (n,).
        **kwargs:
            Score-specific extras matching those passed to ``score``.

        Returns
        -------
        np.ndarray
            Upper prediction bound in observation space, shape (n,).
        """
        ...


class AbsoluteResidualScore:
    """Absolute residual: ``|y - y_hat|``.

    The simplest non-conformity score. Makes no distributional assumptions
    about the base forecaster. Works with any point forecaster.

    Inverting: ``y_hat + s`` gives the upper bound; ``y_hat - s`` the lower.
    For count data, clip the lower bound at zero.

    Parameters
    ----------
    clip_lower:
        If True, ``inverse`` for the lower bound clips at zero. Default True.
        Set False for signed continuous series (e.g. loss ratios).
    """

    def __init__(self, clip_lower: bool = True) -> None:
        self.clip_lower = clip_lower

    def score(self, y: np.ndarray, y_hat: np.ndarray, **kwargs) -> np.ndarray:
        """Compute |y - y_hat|.

        Parameters
        ----------
        y:
            Observed values.
        y_hat:
            Point forecasts.

        Returns
        -------
        np.ndarray
            Absolute residuals.
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        return np.abs(y - y_hat)

    def inverse(
        self,
        s: float | np.ndarray,
        y_hat: np.ndarray,
        upper: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Invert score to prediction bound.

        Parameters
        ----------
        s:
            Score quantile.
        y_hat:
            Point forecasts.
        upper:
            If True, return upper bound (y_hat + s); else lower (y_hat - s).

        Returns
        -------
        np.ndarray
            Prediction interval bound.
        """
        y_hat = np.asarray(y_hat, dtype=float)
        s = np.asarray(s, dtype=float)
        if upper:
            return y_hat + s
        bound = y_hat - s
        if self.clip_lower:
            bound = np.maximum(bound, 0.0)
        return bound


class PoissonPearsonScore:
    """Pearson residual for Poisson base forecaster: ``(y - mu) / sqrt(mu)``.

    Appropriate when the base forecaster returns Poisson mean estimates.
    The Pearson residual has approximate unit variance under the Poisson
    model, so the non-conformity score is more comparable across periods
    with different exposure levels.

    Caveat: near-zero counts make ``sqrt(mu)`` small, inflating the score.
    Add ``min_mu`` for numerical stability.

    Parameters
    ----------
    min_mu:
        Floor on ``mu`` before dividing. Prevents division-by-zero on
        periods with zero expected claims. Default 0.01.
    """

    def __init__(self, min_mu: float = 0.01) -> None:
        if min_mu <= 0:
            raise ValueError("min_mu must be positive.")
        self.min_mu = min_mu

    def score(self, y: np.ndarray, y_hat: np.ndarray, **kwargs) -> np.ndarray:
        """Compute signed Pearson residual.

        Note: returns signed residuals. Conformal methods that need a
        one-sided score should compute ``np.maximum(score, 0)`` or use
        the absolute value. Methods in this library handle this internally.

        Parameters
        ----------
        y:
            Observed claim counts.
        y_hat:
            Predicted Poisson mean (mu).

        Returns
        -------
        np.ndarray
            Signed Pearson residuals (y - mu) / sqrt(mu).
        """
        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(y_hat, dtype=float), self.min_mu)
        return (y - mu) / np.sqrt(mu)

    def inverse(
        self,
        s: float | np.ndarray,
        y_hat: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Invert Pearson score to upper prediction bound.

        Solving ``(y - mu) / sqrt(mu) = s`` for y:
        ``y = mu + s * sqrt(mu)``.

        Parameters
        ----------
        s:
            Score quantile.
        y_hat:
            Predicted Poisson mean.

        Returns
        -------
        np.ndarray
            Upper prediction bound: ``mu + s * sqrt(mu)``.
        """
        mu = np.maximum(np.asarray(y_hat, dtype=float), self.min_mu)
        s = np.asarray(s, dtype=float)
        return mu + s * np.sqrt(mu)


class NegBinomPearsonScore:
    """Pearson residual for Negative Binomial base forecaster.

    Supports NB1 and NB2 parameterisations:

    - **NB2** (most common): Var(Y) = mu + mu^2 / phi. Standard NB in
      statsmodels. Overdispersion grows quadratically with the mean.
    - **NB1**: Var(Y) = mu + mu / phi. Overdispersion grows linearly.

    The score is ``(y - mu) / sqrt(Var(mu, phi))``.

    Parameters
    ----------
    phi:
        Dispersion parameter. If None, must be supplied via ``score()``
        keyword ``phi``. Fitting phi from data is done in the insurance
        wrappers.
    parameterisation:
        ``"NB2"`` or ``"NB1"``. Default ``"NB2"``.
    min_var:
        Floor on the variance before dividing. Default 0.01.
    """

    def __init__(
        self,
        phi: float | None = None,
        parameterisation: str = "NB2",
        min_var: float = 0.01,
    ) -> None:
        if parameterisation not in ("NB1", "NB2"):
            raise ValueError("parameterisation must be 'NB1' or 'NB2'.")
        self.phi = phi
        self.parameterisation = parameterisation
        self.min_var = min_var

    def _variance(self, mu: np.ndarray, phi: float) -> np.ndarray:
        if self.parameterisation == "NB2":
            return mu + mu**2 / phi
        else:  # NB1
            return mu + mu / phi

    def score(
        self, y: np.ndarray, y_hat: np.ndarray, phi: float | None = None, **kwargs
    ) -> np.ndarray:
        """Compute NB Pearson residual.

        Parameters
        ----------
        y:
            Observed counts.
        y_hat:
            Predicted NB mean (mu).
        phi:
            Dispersion parameter. Overrides ``self.phi`` if supplied.

        Returns
        -------
        np.ndarray
            Signed Pearson residuals.
        """
        effective_phi = phi if phi is not None else self.phi
        if effective_phi is None:
            raise ValueError(
                "phi must be supplied either at construction or as a keyword argument."
            )
        y = np.asarray(y, dtype=float)
        mu = np.asarray(y_hat, dtype=float)
        var = np.maximum(self._variance(mu, effective_phi), self.min_var)
        return (y - mu) / np.sqrt(var)

    def inverse(
        self,
        s: float | np.ndarray,
        y_hat: np.ndarray,
        phi: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Invert NB Pearson score to upper prediction bound.

        Solving ``(y - mu) / sqrt(Var) = s`` for y:
        ``y = mu + s * sqrt(Var(mu, phi))``.

        Parameters
        ----------
        s:
            Score quantile.
        y_hat:
            Predicted NB mean.
        phi:
            Dispersion parameter.

        Returns
        -------
        np.ndarray
            Upper prediction bound.
        """
        effective_phi = phi if phi is not None else self.phi
        if effective_phi is None:
            raise ValueError("phi must be supplied.")
        mu = np.asarray(y_hat, dtype=float)
        s = np.asarray(s, dtype=float)
        var = np.maximum(self._variance(mu, effective_phi), self.min_var)
        return mu + s * np.sqrt(var)


class ExposureAdjustedScore:
    """Rate-based non-conformity score: ``y/E - lambda_hat``.

    Divides observed counts by exposure E to obtain an observed rate,
    then subtracts the forecast rate. Useful when exposure varies
    substantially across periods (e.g. policy count changes, premium growth).

    Without exposure adjustment, periods with high exposure dominate the
    calibration set and the intervals are miscalibrated for low-exposure
    periods.

    Parameters
    ----------
    min_exposure:
        Floor on exposure before dividing. Default 1.0 (appropriate when
        E is policy count).
    clip_lower:
        Clip lower bound at zero in ``inverse``. Default True.
    """

    def __init__(self, min_exposure: float = 1.0, clip_lower: bool = True) -> None:
        self.min_exposure = min_exposure
        self.clip_lower = clip_lower

    def score(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        exposure: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute exposure-adjusted residual.

        Parameters
        ----------
        y:
            Observed claim counts.
        y_hat:
            Forecast claim rate (lambda_hat, i.e. expected claims per
            unit exposure).
        exposure:
            Exposure per period. If None, defaults to ones (no adjustment).

        Returns
        -------
        np.ndarray
            Signed rate residuals: y/E - lambda_hat.
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        if exposure is None:
            E = np.ones_like(y)
        else:
            E = np.maximum(np.asarray(exposure, dtype=float), self.min_exposure)
        return y / E - y_hat

    def inverse(
        self,
        s: float | np.ndarray,
        y_hat: np.ndarray,
        exposure: np.ndarray | None = None,
        upper: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Invert to count-space prediction bound.

        Solves ``y/E - lambda_hat = s`` for y: ``y = E * (lambda_hat + s)``.

        Parameters
        ----------
        s:
            Score quantile.
        y_hat:
            Forecast claim rate.
        exposure:
            Exposure per period.
        upper:
            If True, return upper bound; else lower.

        Returns
        -------
        np.ndarray
            Prediction bound in count space.
        """
        y_hat = np.asarray(y_hat, dtype=float)
        s = np.asarray(s, dtype=float)
        if exposure is None:
            E = np.ones_like(y_hat)
        else:
            E = np.maximum(np.asarray(exposure, dtype=float), self.min_exposure)
        if upper:
            return E * (y_hat + s)
        bound = E * (y_hat - s)
        if self.clip_lower:
            bound = np.maximum(bound, 0.0)
        return bound


class LocallyWeightedScore:
    """Locally weighted non-conformity score: ``(y - mu) / sigma_hat``.

    Uses separate estimates of mean and standard deviation (or any
    heteroscedasticity model). This is the most flexible score and
    typically gives the best interval efficiency when ``sigma_hat``
    is well-estimated.

    ``sigma_hat`` can come from:

    - A second-stage model fitted to squared residuals.
    - The running standard deviation of recent residuals.
    - The conditional standard deviation from a GARCH model.
    - A GAM-smoothed variance estimate.

    Parameters
    ----------
    min_sigma:
        Floor on sigma_hat before dividing. Default 1e-6.
    clip_lower:
        Clip lower bound at zero in ``inverse``. Default False
        (loss ratios and severity are continuous).
    """

    def __init__(self, min_sigma: float = 1e-6, clip_lower: bool = False) -> None:
        self.min_sigma = min_sigma
        self.clip_lower = clip_lower

    def score(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        sigma_hat: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute locally weighted residual.

        Parameters
        ----------
        y:
            Observed values.
        y_hat:
            Point forecasts.
        sigma_hat:
            Estimated standard deviations. If None, defaults to ones
            (reducing to absolute residual).

        Returns
        -------
        np.ndarray
            Signed locally weighted residuals.
        """
        y = np.asarray(y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        if sigma_hat is None:
            sigma = np.ones_like(y)
        else:
            sigma = np.maximum(np.asarray(sigma_hat, dtype=float), self.min_sigma)
        return (y - y_hat) / sigma

    def inverse(
        self,
        s: float | np.ndarray,
        y_hat: np.ndarray,
        sigma_hat: np.ndarray | None = None,
        upper: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Invert to observation-space prediction bound.

        Solves ``(y - mu) / sigma = s`` for y: ``y = mu + s * sigma``.

        Parameters
        ----------
        s:
            Score quantile.
        y_hat:
            Point forecasts.
        sigma_hat:
            Estimated standard deviations.
        upper:
            If True, upper bound; else lower.

        Returns
        -------
        np.ndarray
            Prediction bound.
        """
        y_hat = np.asarray(y_hat, dtype=float)
        s = np.asarray(s, dtype=float)
        if sigma_hat is None:
            sigma = np.ones_like(y_hat)
        else:
            sigma = np.maximum(np.asarray(sigma_hat, dtype=float), self.min_sigma)
        if upper:
            return y_hat + s * sigma
        bound = y_hat - s * sigma
        if self.clip_lower:
            bound = np.maximum(bound, 0.0)
        return bound
