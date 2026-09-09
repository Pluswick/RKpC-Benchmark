"""Read-only integrity audit of every completed v2 job and analysis output."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from rat_kp_core.data import sha256_file
from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    ANALYSIS_RESULT_DIR,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    CONFIG_PATH,
    EXPERIMENT_RESULT_DIR,
    MANIFEST_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    STUDY_ROOT,
)
from rat_kp_core.training import PREDICTION_COLUMNS


MANIFESTS = {
    "primary": PRIMARY_JOB_MANIFEST_PATH,
    "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
    "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
}

EXPECTED_ANALYSIS_ROWS = {
    "primary": {
        "all_job_metrics.csv": 993,
        "part1_seed_paired_metrics.csv": 100,
        "part1_context_effects.csv": 20,
        "loto_tissue_effects.csv": 55,
        "loto_inference.csv": 5,
    },
    "ad_excluded": {
        "all_job_metrics.csv": 762,
        "part1_seed_paired_metrics.csv": 100,
        "part1_context_effects.csv": 20,
        "loto_tissue_effects.csv": 55,
        "loto_inference.csv": 5,
    },
    "condition_specific": {
        "all_job_metrics.csv": 762,
        "part1_seed_paired_metrics.csv": 100,
        "part1_context_effects.csv": 20,
        "loto_tissue_effects.csv": 55,
        "loto_inference.csv": 5,
    },
}


def main() -> None:
    config_hash = sha256_file(CONFIG_PATH)
    seen_job_ids = set()
    job_summaries = {}
    expected_complete_dirs = set()

    for analysis_set, manifest_path in MANIFESTS.items():
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", keep_default_na=False)
        prediction_rows = 0
        for job in manifest.itertuples(index=False):
            if job.job_id in seen_job_ids:
                raise RuntimeError(f"Duplicate job ID across manifests: {job.job_id}")
            seen_job_ids.add(job.job_id)
            root = EXPERIMENT_RESULT_DIR / job.analysis_set / job.stage / job.job_id
            expected_complete_dirs.add(root.resolve())
            required = [root / "COMPLETE", root / "run.json", root / "predictions.csv"]
            if not all(path.exists() for path in required) or (root / "FAILED.json").exists():
                raise RuntimeError(f"Incomplete or failed output: {job.job_id}")

            run = json.loads((root / "run.json").read_text(encoding="utf-8"))
            if run.get("status") != "complete" or run["job"]["job_id"] != job.job_id:
                raise RuntimeError(f"Run metadata mismatch: {job.job_id}")
            if run["fixed_config_sha256"] != config_hash:
                raise RuntimeError(f"Config checksum mismatch: {job.job_id}")
            dataset_path = STUDY_ROOT / job.dataset_path
            if run["dataset_sha256"] != sha256_file(dataset_path):
                raise RuntimeError(f"Dataset checksum mismatch: {job.job_id}")

            predictions = pd.read_csv(root / "predictions.csv", encoding="utf-8-sig")
            if list(predictions.columns) != PREDICTION_COLUMNS:
                raise RuntimeError(f"Prediction schema mismatch: {job.job_id}")
            if len(predictions) != int(run["partition_rows"]["test"]):
                raise RuntimeError(f"Prediction/test row mismatch: {job.job_id}")
            if predictions["row_index"].duplicated().any():
                raise RuntimeError(f"Duplicate prediction row_index: {job.job_id}")
            if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all():
                raise RuntimeError(f"Non-finite prediction: {job.job_id}")
            for column in ("job_id", "analysis_set", "stage", "model", "context"):
                if set(predictions[column].astype(str)) != {str(getattr(job, column))}:
                    raise RuntimeError(f"Prediction metadata mismatch ({column}): {job.job_id}")
            prediction_rows += len(predictions)

        job_summaries[analysis_set] = {
            "jobs": len(manifest),
            "prediction_rows": prediction_rows,
            "manifest_sha256": sha256_file(manifest_path),
        }

    observed_complete_dirs = {
        path.parent.resolve() for path in EXPERIMENT_RESULT_DIR.rglob("COMPLETE")
    }
    if observed_complete_dirs != expected_complete_dirs:
        raise RuntimeError("Completed job directories do not exactly match frozen manifests")

    analysis_summaries = {}
    for analysis_set, expected in EXPECTED_ANALYSIS_ROWS.items():
        output = ANALYSIS_RESULT_DIR / analysis_set
        details = {}
        for filename, expected_rows in expected.items():
            path = output / filename
            frame = pd.read_csv(path)
            if len(frame) != expected_rows:
                raise RuntimeError(f"Unexpected analysis rows: {path} ({len(frame)})")
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy(float)).all():
                raise RuntimeError(f"Non-finite analysis value: {path}")
            details[filename] = {
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
        predictions_path = output / "loto_seed_averaged_predictions.csv"
        predictions = pd.read_csv(predictions_path)
        if predictions.empty or not np.isfinite(
            predictions[["y_true", "y_pred", "seed_sd"]].to_numpy(float)
        ).all():
            raise RuntimeError(f"Invalid seed-averaged predictions: {analysis_set}")
        details["loto_seed_averaged_predictions.csv"] = {
            "rows": len(predictions),
            "sha256": sha256_file(predictions_path),
        }
        analysis_summaries[analysis_set] = details

    report = {
        "status": "passed",
        "scope": "completed-result integrity only; no model rerun and no scientific interpretation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_jobs": len(seen_job_ids),
        "failed_jobs": 0,
        "unexpected_complete_directories": 0,
        "fixed_config_sha256": config_hash,
        "job_summaries": job_summaries,
        "analysis_summaries": analysis_summaries,
    }
    path = MANIFEST_DIR / "post_run_integrity_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "completed_jobs": len(seen_job_ids),
        "analysis_sets": list(analysis_summaries),
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
