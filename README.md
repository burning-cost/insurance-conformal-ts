# insurance-conformal-ts

[![PyPI](https://img.shields.io/pypi/v/insurance-conformal-ts)](https://pypi.org/project/insurance-conformal-ts/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-conformal-ts)](https://pypi.org/project/insurance-conformal-ts/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-conformal-ts)](https://pypi.org/project/insurance-conformal-ts/)
[![License](https://img.shields.io/pypi/l/insurance-conformal-ts)](https://pypi.org/project/insurance-conformal-ts/)

Conformal prediction intervals for non-exchangeable insurance claims time series.

**Blog post:** [Your Conformal Intervals Are Wrong When the Claims Series Has Trend](https://burning-cost.github.io/2028/01/15/conformal-prediction-for-non-exchangeable-claims-time-series/)

---

## The problem

Standard conformal prediction gives finite-sample valid prediction intervals. The guarantee requires one thing: exchangeability — that calibration and test points come from the same distribution.

Insurance claims time series violate this. Seasonal loss patterns mean Q4 claims look nothing like Q1. Market-wide hardening shifts the frequency of large losses. IBNR development creates systematic autocorrelation. The test set is never exchangeable with the calibration set.

The naive response is to ignore this and apply standard split conformal anyway. This works approximately when shifts are small, but fails precisely when you need it most: during rapid market dislocation or after a large risk event.

This library implements methods that handle the temporal case properly.

---

## What's inside

### Methods

**ACI** — Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021)

Tracks a running miscoverage level `alpha_t` that adapts based on whether each observation falls inside the prediction interval. If the interval misses, `alpha_t` decreases (widening the next interval); if it covers, `alpha_t` increases. The update is:

```
alpha_{t+1} = alpha_t + gamma * (alpha - 1{y_t not in C_t})
```

Simple, cheap, and effective. This is where to start.

**EnbPI** — Ensemble Batch Prediction Intervals (Xu & Xie, ICML 2021)

Bootstrap ensemble of base forecasters with rolling residual replacement. Handles distribution shift by discarding stale residuals. Requires fitting B forecasters — more expensive than ACI but better when the base forecaster is informative.

**SPCI** — Sequential Predictive Conformal Inference (Xu et al., ICML 2023)

Fits a quantile regression on lagged non-conformity scores to predict the future score distribution. Narrower intervals than EnbPI when the score series is autocorrelated — common in insurance.

**ConformalPID** — PID control for quantile tracking (Angelopoulos et al., NeurIPS 2023)

Proportional + Integral (with saturation) + Derivative control on the coverage error signal. Best theoretical regret bounds.

**MSCP** — Multi-Step Split Conformal Prediction

Horizon-specific calibration for h=1..H. The benchmark winner on sequential multi-step problems. Produces fan charts directly.

### Non-conformity scores

- `AbsoluteResidualScore`: `|y - y_hat|`. No distributional assumptions.
- `PoissonPearsonScore`: `(y - mu) / sqrt(mu)`. For Poisson base models.
- `NegBinomPearsonScore`: NB1/NB2 Pearson residuals with dispersion parameter.
- `ExposureAdjustedScore`: Rate-based score `y/E - lambda_hat`. Use when exposure varies across periods.
- `LocallyWeightedScore`: `(y - mu) / sigma_hat`. Best efficiency when you have a variance model.

### Insurance wrappers

- `ClaimsCountConformal`: Poisson GLM + exposure offset + any sequential method.
- `LossRatioConformal`: Direct loss ratio series.
- `SeverityConformal`: Average claim cost series.

### Diagnostics

- `SequentialCoverageReport`: Rolling coverage, coverage drift (OLS), Kupiec POF test.
- `IntervalWidthReport`: Width over time, widening trend detection.
- `plot_fan_chart`: Multi-step fan chart (matplotlib).

---

## Install

```bash
uv add insurance-conformal-ts
```

For plots:

```bash
uv add "insurance-conformal-ts[plots]"
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-conformal-ts/discussions). Found it useful? A ⭐ helps others find it.

---

## Quickstart

### Single-step intervals (ACI)

```python
import numpy as np
from insurance_conformal_ts import ACI, ClaimsCountConformal, ConstantForecaster

# Monthly claim counts. ACI needs calibration data to produce finite intervals.
# Rule of thumb: training set >= 36 months. On 60 train / 12 test, the default
# burn_in=5 means the first 5 test intervals are (0, inf); from step 6 onward
# the intervals are fully adaptive. Use more training data or a split-conformal
# calibration step to get finite intervals from step 1.
y_train = ...  # shape (60,) — 5 years monthly, minimum recommended
y_test = ...   # shape (12,)

# Option A: ClaimsCountConformal (Poisson GLM + ACI + Pearson score)
ccc = ClaimsCountConformal()
ccc.fit(y_train)
lower, upper = ccc.predict_interval(y_test, alpha=0.1)

report = ccc.coverage_report(y_test, lower, upper)
print(f"Empirical coverage: {report['coverage']:.1%}")  # should be ~90%
print(f"Mean interval width: {report['mean_width']:.1f} claims")

# Option B: roll your own base forecaster (ConstantForecaster as a baseline)
forecaster = ConstantForecaster()
aci = ACI(forecaster, burn_in=5)
aci.fit(y_train)
lower, upper = aci.predict_interval(y_test, alpha=0.1)
```

### Multi-step fan chart (MSCP)

```python
from insurance_conformal_ts import MSCP
from insurance_conformal_ts.nonconformity import AbsoluteResidualScore

mscp = MSCP(my_forecaster, H=12)
mscp.fit(y_train)
mscp.calibrate(y_cal, alpha=0.1)

fan = mscp.predict_fan(alpha=0.1)
# fan[1] = (lower_1step, upper_1step)
# fan[6] = (lower_6step, upper_6step)

from insurance_conformal_ts import plot_fan_chart
plot_fan_chart(y_train, fan, origin_index=len(y_train))
```

### Coverage diagnostics

```python
from insurance_conformal_ts import SequentialCoverageReport, IntervalWidthReport

cov = SequentialCoverageReport(window=12).compute(y_test, lower, upper, alpha=0.1)
print(f"Kupiec p-value: {cov['kupiec_pvalue']:.3f}")  # > 0.05 = valid
print(f"Coverage drift slope: {cov['coverage_drift_slope']:.4f}")  # ~0 = stable

wid = IntervalWidthReport(window=12).compute(lower, upper)
print(f"Median width: {wid['median_width']:.1f}")
```

---

## Bring your own forecaster

Any object with `fit(y, X=None)` and `predict(X=None)` works:

```python
from sklearn.linear_model import PoissonRegressor

class SklearnWrapper:
    def __init__(self):
        self._model = PoissonRegressor()
    def fit(self, y, X=None):
        self._model.fit(X, y)
        return self
    def predict(self, X=None):
        return self._model.predict(X)

from insurance_conformal_ts import ACI, ClaimsCountConformal
from insurance_conformal_ts.nonconformity import PoissonPearsonScore

forecaster = SklearnWrapper()
score = PoissonPearsonScore()
method = ACI(forecaster, score=score, gamma=0.02)
ccc = ClaimsCountConformal(base_forecaster=forecaster, method=method, score=score)
```

---

## Design decisions

**Why ACI as the default?** It has one tuning parameter (`gamma`), no training overhead, and works on any series length. EnbPI and SPCI are better when you have a good base forecaster and enough data to train an ensemble. ConformalPID is better when you want the tightest theoretical guarantees.

**Why horizon-specific calibration in MSCP?** Joint calibration (using a single quantile for all horizons) systematically undercovers at near horizons and overcovers at far horizons. The per-horizon approach costs nothing at inference time and eliminates the bias.

**Why signed Pearson residuals (not absolute)?** Because you want the interval to be asymmetric around the forecast. A Poisson process with mean 10 should have a wider upper tail than lower. The signed score captures this; the absolute score doesn't.

**Why not PyMC or probabilistic programming?** The target user is a UK pricing team running on a laptop or Databricks. Bayesian models require posterior sampling and careful prior specification. Conformal methods require none of this and give finite-sample guarantees instead of asymptotic ones.

---


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_conformal_ts_demo.py).

## References

- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*.
- Xu, C., & Xie, Y. (2021). Conformal prediction interval for dynamic time-series. *ICML 2021*.
- Xu, C., Jiang, Y., & Xie, Y. (2023). Sequential predictive conformal inference for time series. *ICML 2023*.
- Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2023). Conformal PID control for time series prediction. *NeurIPS 2023*.
- arXiv:2601.18509 (2026). Multi-step conformal prediction benchmark.


## Performance

Benchmarked on a synthetic monthly motor claims series with a deliberate structural break. Full script: `benchmarks/run_benchmark.py`.

**DGP:** Poisson counts with seasonal pattern, mild upward trend, and a +20% step shift at test start (simulating market hardening). Base forecaster: constant (training mean) — intentionally simple so conformal coverage correction is the differentiator. Target coverage: 90%.

### Short horizon (24 months test)

60 train, 24 test months. Realistic monitoring window.

| Method | Coverage | Width | Kupiec p |
|--------|----------|-------|----------|
| Target | 0.900 | — | — |
| Naive fixed-width | 0.375 | 60.3 | 0.0000 |
| Split conformal | 0.500 | 86.5 | 0.0000 |
| ACI (this library) | **0.792** | 136.7 | **0.1163** |
| ConformalPID (this library) | 0.625 | 95.8 | 0.0003 |

Even on only 24 test months, ACI achieves statistically valid coverage (Kupiec p=0.12 > 0.05). The static methods are rejected outright.

### Long horizon (60 months test)

60 train, 60 test months. Demonstrates convergence with a mature monitoring series.

| Method | Coverage | Width | Kupiec p |
|--------|----------|-------|----------|
| Target | 0.900 | — | — |
| Naive fixed-width | 0.383 | 60.3 | 0.0000 |
| Split conformal | 0.483 | 86.5 | 0.0000 |
| ACI (this library) | **0.850** | 155.4 | **0.2256** |
| ConformalPID (this library) | 0.667 | 117.3 | 0.0000 |

ACI converges to 85% coverage (Kupiec p=0.23), tracking the target through the structural break. The second-half coverage (months 31–60) reaches 90% exactly, confirming convergence. ConformalPID is more conservative but shows the same improving trend.

**Coverage convergence (ACI):**

| Test window | First half | Second half |
|-------------|-----------|-------------|
| 24 months | 0.667 | 0.917 |
| 60 months | 0.800 | 0.900 |

Static methods stay flat at ~38–50% regardless of horizon length. Adaptive methods close the gap because they respond to observed miscoverage.

The naive fixed interval achieves narrower width (60 vs 155) but at the cost of completely invalid coverage. There is no free lunch: maintaining temporal validity under distribution shift requires adapting interval width.

**When to use:** Any insurance time series where the calibration period differs from the test period — which in practice means any time the model has been in production for more than a quarter. Monthly claims counts, loss ratios, and severity series all exhibit distribution shift that invalidates static conformal intervals.

## Related Libraries

| Library | Description |
|---------|-------------|
| [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) | Cross-sectional conformal prediction — the exchangeable version, for per-risk prediction intervals |
| [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — tracks the drift that violates exchangeability assumptions this library corrects for |
| [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) | Quantile GBMs — parametric alternative for time-series tail risk estimation |
