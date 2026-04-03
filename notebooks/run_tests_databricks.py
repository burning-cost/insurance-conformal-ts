# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-conformal-ts: Run Test Suite
# MAGIC
# MAGIC This notebook installs the package and runs the full pytest suite.
# MAGIC Run on Serverless or any cluster with Python 3.10+.

# COMMAND ----------

# MAGIC %pip install pytest pytest-cov statsmodels scikit-learn scipy numpy pandas -q

# COMMAND ----------

import subprocess, sys, os

# Install the package from the workspace filesystem
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/Repos/insurance-conformal-ts", "-q"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/Repos/insurance-conformal-ts/tests/",
     "-v", "--tb=short", "-q",
     "--no-header"],
    capture_output=True, text=True,
    cwd="/Workspace/Repos/insurance-conformal-ts"
)
print(result.stdout[-10000:])
if result.stderr:
    print("STDERR:", result.stderr[-3000:])
print("Return code:", result.returncode)
