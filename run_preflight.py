"""Run 15 reduced primary integration jobs without retaining performance values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from rat_kp_core.paths import PRIMARY_JOB_MANIFEST_PATH, STUDY_ROOT
from rat_kp_core.training import PREDICTION_COLUMNS, execute_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    jobs = pd.read_csv(PRIMARY_JOB_MANIFEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    selected = jobs[
        (jobs["stage"] == "part1")
        & (jobs["split_type"] == "random")
        & (jobs["split_seed"] == 0)
    ]
    if len(selected) != 15:
        raise RuntimeError(f"Expected 15 preflight jobs, found {len(selected)}")

    checks = []
    temporary_parent = STUDY_ROOT / "results"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preflight_", dir=temporary_parent) as temp:
        root = Path(temp)
        for _, job in selected.iterrows():
            output = execute_job(job, device=args.device, smoke=True, output_root=root)
            predictions = pd.read_csv(output / "predictions.csv", encoding="utf-8-sig")
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            if list(predictions.columns) != PREDICTION_COLUMNS:
                raise RuntimeError(f"Prediction schema mismatch for {job['job_id']}")
            if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all():
                raise RuntimeError(f"Non-finite preflight output for {job['job_id']}")
            if not run.get("smoke_override") or not (output / "COMPLETE").exists():
                raise RuntimeError(f"Completion contract failed for {job['job_id']}")
            checks.append({
                "job_id": job["job_id"],
                "model": job["model"],
                "context": job["context"],
                "prediction_rows": len(predictions),
                "schema_valid": True,
                "finite_predictions": True,
                "parent_group_ids_present": bool(predictions["parent_group_id"].astype(str).ne("").all()),
                "identity_unit_ids_present": bool(predictions["identity_unit_id"].astype(str).ne("").all()),
            })

    output_dir = STUDY_ROOT / "results" / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "passed",
        "scope": "reduced integration only; temporary predictions and metrics deleted",
        "retained_performance_values": False,
        "device": args.device,
        "checks": checks,
    }
    filename = "preflight_manifest.json" if args.device == "cpu" else "preflight_manifest_gpu.json"
    (output_dir / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
