"""
Lambda entry point for the nightly ML batch job.

Thin wrapper only - no ML logic lives here. It runs the existing
scripts/pull_and_check.py and scripts/run_nightly_batch.py in sequence,
as subprocesses, with cwd="/tmp" so their relative "data/..." paths land
somewhere writable. See decisions.md Section 15 / deployment-roadmap.md
Phase 1 for why this shape.
"""
import json
import os
import subprocess
import sys
import boto3

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
WORK_DIR = "/tmp"


def _get_secret(env_var_name: str, ssm_param_name: str) -> str:
    """Prefer an already-set env var (local RIE testing); otherwise fetch
    from Parameter Store (the real deployed path)."""
    value = os.environ.get(env_var_name, "").strip()
    if value:
        return value
    ssm = boto3.client("ssm")
    resp = ssm.get_parameter(Name=ssm_param_name, WithDecryption=True)
    return resp["Parameter"]["Value"]


def _run_step(script_name: str) -> None:
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"--- running {script_name} (cwd={WORK_DIR}) ---")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=WORK_DIR,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} exited with code {result.returncode}")


def handler(event, context):
    os.makedirs(WORK_DIR, exist_ok=True)

    os.environ["TWELVE_DATA_API_KEY"] = _get_secret(
        "TWELVE_DATA_API_KEY", "/ml-batch/TWELVE_DATA_API_KEY"
    )
    os.environ["BATCH_AUTH_TOKEN"] = _get_secret(
        "BATCH_AUTH_TOKEN", "/ml-batch/BATCH_AUTH_TOKEN"
    )

    _run_step("pull_and_check.py")
    _run_step("run_nightly_batch.py")

    payload_path = os.path.join(WORK_DIR, "data", "final_payload.json")
    with open(payload_path) as f:
        payload = json.load(f)

    return {
        "statusCode": 200,
        "predictions_count": len(payload.get("predictions", [])),
    }