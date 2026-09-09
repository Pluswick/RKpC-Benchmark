"""Post-analysis fold-accuracy assessment for the frozen injection study.

This script does not retrain models. It reads the completed primary Part 1 test
predictions and adds interpretable F2/F3/F4 and absolute fold-error summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_injection_results import load_complete_predictions
from rat_kp_fusion.config import load_config
from rat_kp_fusion.jobs import build_plan
from rat_kp_fusion.paths import RESULT_DIR
from rat_kp_core.data import sha256_file


OUTPUT_DIR = RESULT_DIR / "analysis" / "fold_accuracy"
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPLICATES = 10_000
THRESHOLDS = {"f2": 2.0, "f3": 3.0, "f4": 4.0}


def _bootstrap_mean_ci(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def _add_fold_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["abs_log10_error"] = np.abs(
        result["y_pred"].to_numpy(float) - result["y_true"].to_numpy(float)
    )
    result["absolute_fold_error"] = np.power(10.0, result["abs_log10_error"])
    for name, threshold in THRESHOLDS.items():
        result[name] = result["abs_log10_error"] <= np.log10(threshold) + 1e-15
    return result


def _metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    abs_errors = frame["abs_log10_error"].to_numpy(float)
    fold_errors = frame["absolute_fold_error"].to_numpy(float)
    return {
        "n_rows": int(len(frame)),
        "n_parent_groups": int(frame["parent_group_id"].nunique()),
        "f2": float(frame["f2"].mean()),
        "f3": float(frame["f3"].mean()),
        "f4": float(frame["f4"].mean()),
        "mae_log10": float(abs_errors.mean()),
        "geometric_mean_fold_error": float(10.0 ** abs_errors.mean()),
        "median_absolute_fold_error": float(np.median(fold_errors)),
        "p90_absolute_fold_error": float(np.quantile(fold_errors, 0.90)),
    }


def make_job_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "job_id", "model", "condition", "split_type", "split_seed", "training_seed"
    ]
    rows = []
    for key, frame in predictions.groupby(grouping, sort=True):
        row = dict(zip(grouping, key))
        row.update(_metric_row(frame))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 400 or set(result.groupby(["model", "condition", "split_type"]).size()) != {10}:
        raise RuntimeError("Primary Part 1 must contain 40 combinations x 10 split seeds")
    return result


def _parent_macro_summary(
    predictions: pd.DataFrame,
    grouping: list[str],
) -> pd.DataFrame:
    parent_metrics = (
        predictions.groupby(grouping + ["parent_group_id"], as_index=False, sort=True)
        .agg(
            f2=("f2", "mean"),
            f3=("f3", "mean"),
            f4=("f4", "mean"),
            n_prediction_rows=("row_index", "size"),
        )
    )
    rows = []
    for group_index, (key, frame) in enumerate(parent_metrics.groupby(grouping, sort=True)):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(grouping, key))
        row["n_unique_parent_groups"] = int(frame["parent_group_id"].nunique())
        for metric_index, metric in enumerate(THRESHOLDS):
            values = frame[metric].to_numpy(float)
            low, high = _bootstrap_mean_ci(values, seed_offset=1000 + group_index * 10 + metric_index)
            row[f"parent_macro_{metric}"] = float(values.mean())
            row[f"parent_macro_{metric}_ci95_low"] = low
            row[f"parent_macro_{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def make_combination_summary(predictions: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    grouping = ["split_type", "model", "condition"]
    rows = []
    for group_index, (key, frame) in enumerate(jobs.groupby(grouping, sort=True)):
        row = dict(zip(grouping, key))
        row["n_split_seeds"] = int(frame["split_seed"].nunique())
        for metric_index, metric in enumerate(["f2", "f3", "f4"]):
            values = frame[metric].to_numpy(float)
            low, high = _bootstrap_mean_ci(values, seed_offset=group_index * 10 + metric_index)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = float(values.std(ddof=1))
            row[f"split_ci95_{metric}_low"] = low
            row[f"split_ci95_{metric}_high"] = high
        for metric in [
            "geometric_mean_fold_error", "median_absolute_fold_error",
            "p90_absolute_fold_error", "mae_log10",
        ]:
            row[f"mean_job_{metric}"] = float(frame[metric].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    parent = _parent_macro_summary(predictions, grouping)
    return summary.merge(parent, on=grouping, validate="one_to_one")


def make_tissue_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    job_grouping = ["split_type", "model", "condition", "split_seed", "Tissue"]
    job_tissue_rows = []
    for key, frame in predictions.groupby(job_grouping, sort=True):
        row = dict(zip(job_grouping, key))
        row.update(_metric_row(frame))
        job_tissue_rows.append(row)
    job_tissues = pd.DataFrame(job_tissue_rows)

    grouping = ["split_type", "model", "condition", "Tissue"]
    rows = []
    for group_index, (key, frame) in enumerate(job_tissues.groupby(grouping, sort=True)):
        row = dict(zip(grouping, key))
        row["n_split_seeds"] = int(frame["split_seed"].nunique())
        for metric_index, metric in enumerate(THRESHOLDS):
            values = frame[metric].to_numpy(float)
            low, high = _bootstrap_mean_ci(values, seed_offset=5000 + group_index * 10 + metric_index)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"split_ci95_{metric}_low"] = low
            row[f"split_ci95_{metric}_high"] = high
        row["mean_job_median_absolute_fold_error"] = float(
            frame["median_absolute_fold_error"].mean()
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    parent = _parent_macro_summary(predictions, grouping)
    return summary.merge(parent, on=grouping, validate="one_to_one")


def main() -> None:
    load_config(require_frozen=True)
    plan = build_plan()
    primary_part1 = plan[(plan["analysis_set"] == "primary") & (plan["stage"] == "part1")].copy()
    if len(primary_part1) != 400:
        raise RuntimeError(f"Expected 400 primary Part 1 jobs, found {len(primary_part1)}")
    predictions = _add_fold_columns(load_complete_predictions(primary_part1))
    if not np.isfinite(predictions[["y_true", "y_pred", "absolute_fold_error"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite prediction or fold-error value")

    jobs = make_job_metrics(predictions)
    combinations = make_combination_summary(predictions, jobs)
    tissues = make_tissue_summary(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fold_accuracy_job_metrics.csv": jobs,
        "fold_accuracy_summary.csv": combinations,
        "fold_accuracy_tissue_summary.csv": tissues,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "complete",
        "scope": "frozen primary Part 1 test predictions; no retraining",
        "fold_definitions": {
            name: f"abs(y_pred-y_true) <= log10({threshold:g})"
            for name, threshold in THRESHOLDS.items()
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_SEED,
            "split_ci": "bootstrap mean across 10 split-seed job metrics",
            "parent_ci": "bootstrap mean across parent-group-level accuracies after pooling test appearances",
        },
        "prediction_rows": int(len(predictions)),
        "jobs": int(len(jobs)),
        "outputs": {
            filename: {"rows": int(len(frame)), "sha256": sha256_file(OUTPUT_DIR / filename)}
            for filename, frame in outputs.items()
        },
    }
    manifest_path = OUTPUT_DIR / "fold_accuracy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    top = combinations.sort_values(["split_type", "mean_f4"], ascending=[True, False]).groupby(
        "split_type", as_index=False
    ).head(5)
    print(json.dumps({
        "status": "complete",
        "prediction_rows": len(predictions),
        "jobs": len(jobs),
        "combination_rows": len(combinations),
        "tissue_rows": len(tissues),
        "top_by_f4": top[[
            "split_type", "model", "condition", "mean_f2", "mean_f3", "mean_f4",
            "mean_job_median_absolute_fold_error", "parent_macro_f4",
        ]].to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
