from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rat_kp_dvi.config import load_config
from rat_kp_dvi.jobs import build_full_context_jobs
from rat_kp_dvi.paths import ANALYSIS_DIR, EXPERIMENT_DIR, RESULT_DIR


def main() -> None:
    cfg = load_config(); failures = []; checks = {}
    volumes = cfg["tissue_volumes_ml"]
    checks["tissue_count"] = len(volumes)
    checks["tissue_volume_sum_ml"] = sum(volumes.values())
    checks["blood_volume_L_per_kg"] = cfg["blood_volume_ml"] / cfg["body_weight_g"]
    if len(volumes) != 11 or any(value <= 0 for value in volumes.values()):
        failures.append("Invalid tissue-volume table")
    jobs = build_full_context_jobs()
    parent_jobs = jobs[jobs["split_type"].eq("parent_group")]
    run_records = []
    for row in parent_jobs.itertuples(index=False):
        output = EXPERIMENT_DIR / row.split_type / row.job_id
        if not (output / "COMPLETE").exists():
            failures.append(f"Missing full-grid job: {row.job_id}")
            continue
        run = json.loads((output / "run.json").read_text(encoding="utf-8")); run_records.append(run)
        grid = pd.read_csv(output / "full_grid_predictions.csv", encoding="utf-8-sig")
        counts = grid.groupby("identity_unit_id")["Tissue"].nunique()
        if not counts.eq(11).all() or not np.isfinite(grid["y_pred"]).all():
            failures.append(f"Invalid full grid: {row.job_id}")
    checks["complete_parent_full_grid_jobs"] = len(run_records)
    checks["max_retraining_difference_log10"] = max(record["max_abs_retraining_difference_log10"] for record in run_records)
    if checks["max_retraining_difference_log10"] > 1e-4:
        failures.append("Parent GINE retraining did not reproduce canonical predictions")
    required = [
        "observed_panel_identity_averaged.csv", "vss_propagation_metrics.csv",
        "observed_panel_paired_context_effects.csv", "full_grid_identity_averaged.csv",
        "complete_11_tissue_exploratory.csv", "literature_panel_vss_metrics.csv",
        "additive_reconstruction_audit.csv",
    ]
    for name in required:
        if not (ANALYSIS_DIR / name).exists(): failures.append(f"Missing analysis output: {name}")
    observed = pd.read_csv(ANALYSIS_DIR / "observed_panel_identity_averaged.csv", encoding="utf-8-sig")
    panel_counts = observed.groupby(["split_type", "identity_unit_id"])["method"].nunique()
    if not panel_counts.eq(3).all(): failures.append("Observed panels are not matched across three comparators")
    complete = pd.read_csv(ANALYSIS_DIR / "complete_11_tissue_exploratory.csv", encoding="utf-8-sig")
    checks["complete_11_tissue_identity_units"] = complete["identity_unit_id"].nunique()
    if checks["complete_11_tissue_identity_units"] != 5: failures.append("Unexpected complete-profile count")
    recon = pd.read_csv(ANALYSIS_DIR / "additive_reconstruction_audit.csv", encoding="utf-8-sig")
    checks["max_additive_reconstruction_residual_log10"] = recon.iloc[:, 1].max()
    if checks["max_additive_reconstruction_residual_log10"] > 1e-4:
        failures.append("Additive two-way reconstruction tolerance failed")
    report = {"status": "passed" if not failures else "failed", "checks": checks, "failures": failures}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "VSS_OUTPUT_PROPAGATION_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
