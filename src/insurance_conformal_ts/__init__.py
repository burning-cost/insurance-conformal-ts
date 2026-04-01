"""
insurance-conformal-ts
======================

Conformal prediction intervals for non-exchangeable insurance claims time series.

Standard conformal prediction assumes exchangeability — that the calibration
and test points are drawn i.i.d. from the same distribution. Insurance claims
time series violate this: seasonality, trend, distribution shift, and serial
correlation all break the exchangeability assumption.

This library implements methods that handle the temporal case:

- **ACI** (Gibbs & Candès 2021): Adaptive Conformal Inference. Online quantile
  tracking with a learning rate that adapts alpha_t based on coverage errors.
  Simple and effective; the starting point for most practitioners.

- **EnbPI** (Xu & Xie 2021): Ensemble Batch Prediction Intervals. Bootstrap
  ensemble of base forecasters with rolling residual calibration.

- **SPCI** (Xu et al. 2023): Sequential Predictive Conformal Inference. Uses
  quantile regression on lagged residuals to predict future non-conformity
  scores, giving narrower intervals than EnbPI.

- **ConformalPID** (Angelopoulos et al. NeurIPS 2023): PID controller for
  quantile tracking. Best theoretical guarantees.

- **WeightedConformalPredictor** (WCP): Weighted Conformal Prediction with
  exponential decay. Re-weights calibration scores so recent observations
  matter more. Complementary to ACI and EnbPI; use for slow distribution drift.

- **MSCP**: Multi-Step Split Conformal Prediction. Horizon-specific calibration
  for h=1..H.

Insurance-specific wrappers handle exposure offsets, Poisson/NB base
forecasters, and UK reporting standards.
"""

from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    ExposureAdjustedScore,
    LocallyWeightedScore,
    NegBinomPearsonScore,
    NonConformityScore,
    PoissonPearsonScore,
)
from insurance_conformal_ts.methods import (
    ACI,
    BaseForecaster,
    ConstantForecaster,
    EnbPI,
    ConformalPID,
    MeanForecaster,
    SPCI,
    WeightedConformalPredictor,
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
    plot_fan_chart,
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-conformal-ts")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
__all__ = [
    # Non-conformity scores
    "NonConformityScore",
    "AbsoluteResidualScore",
    "PoissonPearsonScore",
    "NegBinomPearsonScore",
    "ExposureAdjustedScore",
    "LocallyWeightedScore",
    # Methods
    "ACI",
    "EnbPI",
    "SPCI",
    "ConformalPID",
    "WeightedConformalPredictor",
    # Built-in forecasters
    "BaseForecaster",
    "ConstantForecaster",
    "MeanForecaster",
    "MSCP",
    # Insurance wrappers
    "ClaimsCountConformal",
    "LossRatioConformal",
    "SeverityConformal",
    # Diagnostics
    "SequentialCoverageReport",
    "IntervalWidthReport",
    "plot_fan_chart",
]
