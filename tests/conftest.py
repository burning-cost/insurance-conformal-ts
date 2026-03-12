"""
conftest.py
===========

Shared fixtures for the insurance-conformal-ts test suite.

All synthetic data uses a fixed seed so tests are reproducible. The data
generating processes (DGPs) are kept simple and well-specified so we can
verify theoretical properties (coverage, interval width, adaptation) without
relying on approximations.
"""

from __future__ import annotations

import numpy as np
import pytest


RNG_SEED = 42


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def poisson_series(rng):
    """Stationary Poisson time series, lambda=10, n=500."""
    n = 500
    lam = 10.0
    y = rng.poisson(lam, size=n).astype(float)
    return y, lam


@pytest.fixture(scope="session")
def poisson_series_with_shift(rng):
    """Poisson series with a mean shift at t=300 (lambda: 10 -> 20)."""
    n = 500
    y = np.concatenate([
        rng.poisson(10.0, size=300).astype(float),
        rng.poisson(20.0, size=200).astype(float),
    ])
    return y


@pytest.fixture(scope="session")
def poisson_series_with_exposure(rng):
    """Poisson series with varying exposure. Rate lambda=0.15 per unit."""
    n = 500
    rate = 0.15
    exposure = rng.uniform(50, 200, size=n)
    mu = rate * exposure
    y = rng.poisson(mu).astype(float)
    return y, exposure, rate


@pytest.fixture(scope="session")
def loss_ratio_series(rng):
    """Continuous loss ratio series with mild trend and noise."""
    n = 400
    trend = np.linspace(0.6, 0.75, n)
    noise = rng.normal(0, 0.05, size=n)
    y = trend + noise
    return y


@pytest.fixture(scope="session")
def severity_series(rng):
    """Gamma severity series (mean=2000, CV=0.5)."""
    n = 400
    mu = 2000.0
    cv = 0.5
    shape = 1 / cv**2  # = 4
    scale = mu / shape
    y = rng.gamma(shape, scale, size=n)
    return y


class ConstantForecaster:
    """Forecaster that always predicts the training mean.

    Simple enough to reason about analytically; valid for testing
    conformal coverage properties on stationary series.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> "ConstantForecaster":
        self._mean = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        n = len(X) if X is not None else 1
        return np.full(n, self._mean)


class PoissonMeanForecaster:
    """Forecaster that predicts the training Poisson mean."""

    def __init__(self) -> None:
        self._mu: float = 1.0

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> "PoissonMeanForecaster":
        self._mu = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        n = len(X) if X is not None else 1
        return np.full(n, self._mu)


@pytest.fixture
def constant_forecaster():
    return ConstantForecaster()


@pytest.fixture
def poisson_forecaster():
    return PoissonMeanForecaster()
