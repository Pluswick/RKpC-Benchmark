"""One-command, resume-safe execution of all frozen v2 jobs and analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from rat_kp_core.data import sha256_file
from rat_kp_core.paths import (
    AD_SENSITIVITY_DATA_PATH,
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_DATA_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    CONFIG_PATH,
    LONG_DATA_PATH,
    MANIFEST_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    STUDY_ROOT,
)


def verify_frozen_inputs() -> dict:
    """Verify the pre-experiment freeze without rerunning fail-closed validation."""
    validation_path = MANIFEST_DIR / "pre_experiment_validation_v2.json"
    if not validation_path.exists():
        raise RuntimeError("Missing pre_experiment_validation_v2.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise RuntimeError("Pre-experiment validation did not pass")

    if sha256_file(CONFIG_PATH) != validation["fixed_config_sha256"]:
        raise RuntimeError("Frozen model config checksum changed")

    datasets = {
        "primary": LONG_DATA_PATH,
        "ad_excluded": AD_SENSITIVITY_DATA_PATH,
        "condition_specific": CONDITION_SENSITIVITY_DATA_PATH,
    }
    for name, path in datasets.items():
        expected = validation["data_contracts"][name]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen {name} dataset checksum changed")

    job_manifests = {
        "primary": PRIMARY_JOB_MANIFEST_PATH,
        "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
        "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    }
    for name, path in job_manifests.items():
        expected = validation["job_manifest_hashes"][name]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen {name} job manifest checksum changed")

    for filename, device in (
        ("preflight_manifest.json", "cpu"),
        ("preflight_manifest_gpu.json", "gpu"),
    ):
        path = STUDY_ROOT / "results" / "preflight" / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "passed"
            or report.get("device") != device
            or report.get("retained_performance_values") is not False
            or len(report.get("checks", [])) != 15
        ):
            raise RuntimeError(f"Invalid {device} preflight manifest")
    return validation


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=STUDY_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation = verify_frozen_inputs()
    experiment_command = [
        sys.executable,
        str(STUDY_ROOT / "run_experiments.py"),
        "--analysis-set",
        "all",
        "--device",
        args.device,
    ]
    if args.dry_run:
        experiment_command.append("--dry-run")
    run(experiment_command)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_complete",
            "scope": "all frozen primary and sensitivity jobs; analysis not executed",
        }, indent=2))
        return

    run([sys.executable, str(STUDY_ROOT / "check_progress.py")])
    run([
        sys.executable,
        str(STUDY_ROOT / "analyze_results.py"),
        "--analysis-set",
        "all",
    ])

    report = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": "run_full_study.py",
        "device": args.device,
        "python_executable": sys.executable,
        "fixed_config_sha256": validation["fixed_config_sha256"],
        "job_counts": validation["job_counts"],
        "analysis_sets": ["primary", "ad_excluded", "condition_specific"],
    }
    (MANIFEST_DIR / "full_study_run_complete.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
