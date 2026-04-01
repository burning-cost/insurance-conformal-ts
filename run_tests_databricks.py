"""
Submit the insurance-conformal-ts test suite to Databricks via the Jobs API.

Uploads the project to the workspace, then runs pytest via a one-time job
using serverless compute.
"""

import os
import sys
import time
import base64
import pathlib

# Load credentials
with open(os.path.expanduser("~/.config/burning-cost/databricks.env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/insurance-conformal-ts-tests"
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

print(f"Uploading project from {LOCAL_ROOT} to {WORKSPACE_PATH}")

project_remote = f"{WORKSPACE_PATH}/project"

# Upload source files, excluding generated/cache directories
SKIP_PARTS = {".venv", "__pycache__", ".git", "dist"}

for p in pathlib.Path(LOCAL_ROOT).rglob("*.py"):
    rel = p.relative_to(LOCAL_ROOT)
    # Skip .venv, __pycache__, dist, and egg-info dirs
    if any(
        part in SKIP_PARTS or part.endswith(".egg-info")
        for part in rel.parts[:-1]
    ):
        continue
    remote = f"{project_remote}/{rel}"
    with open(str(p), "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()
    try:
        w.workspace.import_(
            path=remote,
            content=content_b64,
            format=ImportFormat.AUTO,
            overwrite=True,
        )
        print(f"  Uploaded: {rel}")
    except Exception as e:
        print(f"  Failed {rel}: {e}")

# Upload pyproject.toml
toml_path = os.path.join(LOCAL_ROOT, "pyproject.toml")
with open(toml_path, "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()
try:
    w.workspace.import_(
        path=f"{project_remote}/pyproject.toml",
        content=content_b64,
        format=ImportFormat.AUTO,
        overwrite=True,
    )
    print("  Uploaded: pyproject.toml")
except Exception as e:
    print(f"  Failed pyproject.toml: {e}")

# Create and upload test runner notebook (proper Databricks source format)
notebook_content = """\
# Databricks notebook source
# MAGIC %pip install statsmodels scikit-learn scipy pandas numpy pytest pytest-cov

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-conformal-ts-tests/project"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print(result.stderr[-1000:])
    raise Exception("pip install failed")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-conformal-ts-tests/project/tests",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-conformal-ts-tests/project"
)
print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
if result.stderr:
    print(result.stderr[-1000:])

if result.returncode != 0:
    raise Exception(f"pytest failed with return code {result.returncode}")
else:
    print("\\n=== ALL TESTS PASSED ===")
"""

runner_path = f"{WORKSPACE_PATH}/run_tests"
nb_b64 = base64.b64encode(notebook_content.encode()).decode()
try:
    w.workspace.import_(
        path=runner_path,
        content=nb_b64,
        format=ImportFormat.SOURCE,
        overwrite=True,
        language=Language.PYTHON,
    )
    print(f"  Uploaded notebook: {runner_path}")
except Exception as e:
    print(f"  Failed notebook: {e}")
    sys.exit(1)

# Submit as serverless job
print("\nSubmitting test job to Databricks (serverless)...")
run = w.jobs.submit(
    run_name="insurance-conformal-ts-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=runner_path,
                base_parameters={},
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(
                client="2",
                dependencies=[
                    "statsmodels",
                    "scikit-learn",
                    "scipy",
                    "pandas",
                    "numpy",
                    "pytest",
                    "pytest-cov",
                ],
            ),
        )
    ],
)

run_id = run.run_id
print(f"Job run submitted: run_id={run_id}")
print(f"Monitor at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

# Poll for completion
print("\nWaiting for job to complete...")
while True:
    status = w.jobs.get_run(run_id=run_id)
    state = status.state.life_cycle_state.value
    result_state = status.state.result_state
    print(f"  State: {state}", end="")
    if result_state:
        print(f"  Result: {result_state.value}", end="")
    print()

    if state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

# Get output
if result_state and result_state.value == "SUCCESS":
    print("\n=== JOB SUCCEEDED ===")
    try:
        output = w.jobs.get_run_output(run_id=run_id)
        if output.notebook_output:
            print(output.notebook_output.result)
    except Exception as e:
        print(f"Could not fetch output: {e}")
else:
    print(f"\n=== JOB FAILED: {result_state} ===")
    try:
        output = w.jobs.get_run_output(run_id=run_id)
        if output.error:
            print(f"Error: {output.error}")
        if output.notebook_output:
            print(output.notebook_output.result)
    except Exception as e:
        print(f"Could not fetch output: {e}")
    sys.exit(1)
