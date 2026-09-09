"""Post-run integrity audit for the isolated additional context analyses."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from rat_kp_core.data import sha256_file
from rat_kp_core.training import load_job_frames
from rat_kp_controls.jobs import build_plan
from rat_kp_controls.paths import (
    ANALYSIS_DIR,
    EXPERIMENT_RESULT_DIR,
    MANIFEST_DIR,
    STUDY_ROOT,
)


def main() -> None:
    lock_path = MANIFEST_DIR / "execution_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_drift = []
    for relative, expected in lock["training_source_sha256"].items():
        path = STUDY_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            source_drift.append({
                "path": relative, "expected_sha256": expected, "observed_sha256": observed
            })
    plan = build_plan()
    failures = []
    prediction_rows = 0
    runtimes = []
    for row in plan.itertuples(index=False):
        output = EXPERIMENT_RESULT_DIR / row.analysis / row.job_id
        complete = output / "COMPLETE"
        run_path = output / "run.json"
        prediction_path = output / "predictions.csv"
        if not complete.exists() or not run_path.exists() or not prediction_path.exists():
            failures.append({"job_id": row.job_id, "reason": "missing_output"})
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("job", {}).get("job_id") != row.job_id:
            failures.append({"job_id": row.job_id, "reason": "run_job_id_mismatch"})
            continue
        if run.get("additional_config_sha256") != lock["training_source_sha256"][
            "manuscript/ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json"
        ]:
            failures.append({"job_id": row.job_id, "reason": "config_hash_mismatch"})
            continue
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
        frames = load_job_frames(pd.Series(row._asdict()))
        if (
            len(predictions) != len(frames["test"])
            or predictions.row_index.tolist() != frames["test"].row_index.tolist()
            or not np.allclose(predictions.y_true, frames["test"].log10Kp)
            or not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all()
        ):
            failures.append({"job_id": row.job_id, "reason": "prediction_contract_failure"})
            continue
        prediction_rows += len(predictions)
        runtimes.append(float(run["runtime_seconds"]))
    quarantine = (
        EXPERIMENT_RESULT_DIR.parent / "quarantine_wrong_gpu_20260827_1015"
    )
    report = {
        "status": "passed" if not source_drift and not failures else "failed",
        "planned_jobs": len(plan),
        "complete_valid_jobs": len(plan) - len(failures),
        "prediction_rows": prediction_rows,
        "runtime_seconds_sum": float(np.sum(runtimes)),
        "source_drift": source_drift,
        "job_failures": failures,
        "quarantine_exists": quarantine.exists(),
        "quarantine_excluded_from_experiment_root": not str(quarantine).startswith(
            str(EXPERIMENT_RESULT_DIR) + "\\"
        ),
        "analysis_manifest_present": (ANALYSIS_DIR / "analysis_manifest.json").exists(),
    }
    output = MANIFEST_DIR / "post_run_integrity_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
