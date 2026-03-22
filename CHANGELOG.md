# Changelog

## v0.1.0 (2026-03-22) [unreleased]
- feat: add Databricks benchmark notebook
- fix: add missing [project.urls] section to pyproject.toml
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.0 (2026-03-21)
- docs: replace pip install with uv add in README
- Add 60-month convergence scenario to benchmark
- Add blog post link and community CTA to README
- fix: update license badge from BSD-3 to MIT
- Add MIT license
- Add benchmarks/run_benchmark.py and update Performance section with real numbers
- Fix batch 11 audit issues: README import, ACI cold-start, EnbPI kw_i bug, ConformalPID error signal
- Add PyPI classifiers for financial/insurance audience
- fix: add tests/__init__.py so CI can resolve `from tests.conftest` imports
- fix: numpy 2.x array-to-scalar assignment in predict_interval
- pin statsmodels>=0.14.5 for scipy compat
- Add shields.io badges for consistency
- docs: add Databricks notebook link
- docs: add Performance section to README
- Fix prediction interval lower bounds and score inversion
- Initial implementation: insurance-conformal-ts v0.1.0

