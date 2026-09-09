"""Analyze completed v2 primary or data-sensitivity manifests without retraining."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from rat_kp_core.jobs import MODELS
from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    ANALYSIS_RESULT_DIR,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    EXPERIMENT_RESULT_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
)
from rat_kp_core.statistics import (
    BOOTSTRAP_REPLICATES,
    benjamini_hochberg,
    exact_sign_flip_pvalue,
    hierarchical_loto_ci,
    percentile_ci,
)


MANIFESTS = {
    "primary": PRIMARY_JOB_MANIFEST_PATH,
    "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
    "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
}


def collect(manifest_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", keep_default_na=False)
    metric_rows, predictions, missing = [], [], []
    for job in manifest.itertuples(index=False):
        root = EXPERIMENT_RESULT_DIR / job.analysis_set / job.stage / job.job_id
        if not (root / "COMPLETE").exists():
            missing.append(job.job_id)
            continue
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        metric_rows.append({
            **job._asdict(),
            **run["test_metrics"],
            "runtime_seconds": run["runtime_seconds"],
        })
        predictions.append(pd.read_csv(root / "predictions.csv", encoding="utf-8-sig"))
    if missing:
        raise RuntimeError(
            f"Cannot analyze incomplete manifest; missing {len(missing)} jobs, e.g. {missing[:3]}"
        )
    return pd.DataFrame(metric_rows), pd.concat(predictions, ignore_index=True)


def part1_analysis(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    part1 = metrics[metrics["stage"] == "part1"].copy()
    keys = ["model", "split_type", "split_seed"]
    pivot = part1.pivot(index=keys, columns="context", values="rmse_log10").reset_index()
    required = {"structure_only", "tissue_onehot", "physiology"}
    if required - set(pivot.columns):
        raise RuntimeError(f"Missing Part 1 contexts: {sorted(required - set(pivot.columns))}")
    pivot["delta_onehot"] = pivot["structure_only"] - pivot["tissue_onehot"]
    pivot["delta_physiology"] = pivot["structure_only"] - pivot["physiology"]

    rows = []
    for (model, split_type), group in pivot.groupby(["model", "split_type"]):
        if len(group) != 10:
            raise RuntimeError(f"Expected 10 paired seeds for {model}/{split_type}")
        for context, column in (
            ("tissue_onehot", "delta_onehot"),
            ("physiology", "delta_physiology"),
        ):
            values = group[column].to_numpy(float)
            low, high = percentile_ci(values)
            rows.append({
                "model": model,
                "split_type": split_type,
                "context": context,
                "mean_delta_rmse": values.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "seed_sd": values.std(ddof=1),
                "n_split_seeds": len(values),
                "ci_method": f"paired split-seed bootstrap, {BOOTSTRAP_REPLICATES} replicates",
            })
    return pivot, pd.DataFrame(rows)


def _seed_average_loto(predictions: pd.DataFrame) -> pd.DataFrame:
    loto = predictions[predictions["stage"] == "loto"].copy()
    keys = [
        "model",
        "context",
        "heldout_tissue",
        "row_index",
        "parent_group_id",
        "identity_unit_id",
        "Drug",
        "SMILES",
        "Tissue",
        "y_true",
    ]
    averaged = loto.groupby(keys, as_index=False).agg(
        y_pred=("y_pred", "mean"),
        seed_sd=("y_pred", "std"),
        n_training_seeds=("training_seed", "nunique"),
    )
    averaged["seed_sd"] = averaged["seed_sd"].fillna(0.0)
    for model in MODELS:
        expected = 1 if model == "linear_svr" else 5
        observed = averaged.loc[averaged["model"] == model, "n_training_seeds"]
        if observed.empty or not (observed == expected).all():
            raise RuntimeError(f"LOTO seed-count contract failed for {model}; expected {expected}")
    return averaged


def loto_analysis(
    predictions: pd.DataFrame,
    *,
    primary: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    averaged = _seed_average_loto(predictions)
    tissue_metrics = (
        averaged.assign(squared_error=lambda x: (x.y_pred - x.y_true) ** 2)
        .groupby(["model", "context", "heldout_tissue"], as_index=False)
        .agg(mse=("squared_error", "mean"), n_rows=("row_index", "size"))
    )
    tissue_metrics["rmse_log10"] = np.sqrt(tissue_metrics["mse"])
    pivot = tissue_metrics.pivot(
        index=["model", "heldout_tissue", "n_rows"],
        columns="context",
        values="rmse_log10",
    ).reset_index()
    required = {"structure_only", "physiology"}
    if required - set(pivot.columns):
        raise RuntimeError(f"Missing LOTO contexts: {sorted(required - set(pivot.columns))}")
    pivot["delta_physiology"] = pivot["structure_only"] - pivot["physiology"]
    if "tissue_onehot" in pivot.columns:
        pivot["delta_unknown_onehot_control"] = pivot["structure_only"] - pivot["tissue_onehot"]

    inference = []
    for model in MODELS:
        group = pivot[pivot["model"] == model].sort_values("heldout_tissue")
        if len(group) != 11:
            raise RuntimeError(f"Expected 11 LOTO tissue deltas for {model}")
        deltas = group["delta_physiology"].to_numpy(float)
        model_predictions = averaged[averaged["model"] == model]
        tissue_rows = {}
        for tissue in sorted(group["heldout_tissue"]):
            tissue_frame = model_predictions[model_predictions["heldout_tissue"] == tissue]
            wide = tissue_frame.pivot(
                index=["row_index", "y_true"], columns="context", values="y_pred"
            ).reset_index()
            tissue_rows[tissue] = (
                wide["y_true"].to_numpy(float),
                wide["structure_only"].to_numpy(float),
                wide["physiology"].to_numpy(float),
            )
        low, high = hierarchical_loto_ci(tissue_rows)
        inference.append({
            "model": model,
            "mean_delta_rmse": deltas.mean(),
            "ci95_low": low,
            "ci95_high": high,
            "raw_p_value": exact_sign_flip_pvalue(deltas),
            "n_tissues": len(deltas),
            "ci_method": f"tissue/compound hierarchical bootstrap, {BOOTSTRAP_REPLICATES} replicates",
            "test_method": "two-sided exact sign-flip over 11 tissue deltas",
            "inference_scope": (
                "prespecified primary FDR family" if primary else "exploratory robustness family"
            ),
        })
    inference = pd.DataFrame(inference)
    fdr_column = "bh_fdr_p_value" if primary else "exploratory_bh_fdr_p_value"
    inference[fdr_column] = benjamini_hochberg(inference["raw_p_value"].to_numpy())
    return averaged, pivot, inference


def analyze_one(name: str) -> dict:
    metrics, predictions = collect(MANIFESTS[name])
    output = ANALYSIS_RESULT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "all_job_metrics.csv", index=False)
    seed_metrics, part1_summary = part1_analysis(metrics)
    seed_metrics.to_csv(output / "part1_seed_paired_metrics.csv", index=False)
    part1_summary.to_csv(output / "part1_context_effects.csv", index=False)
    averaged, tissue_effects, inference = loto_analysis(predictions, primary=name == "primary")
    averaged.to_csv(output / "loto_seed_averaged_predictions.csv", index=False)
    tissue_effects.to_csv(output / "loto_tissue_effects.csv", index=False)
    inference.to_csv(output / "loto_inference.csv", index=False)
    return {"analysis_set": name, "jobs": len(metrics), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-set",
        choices=["primary", "ad_excluded", "condition_specific", "all"],
        default="primary",
    )
    args = parser.parse_args()
    selected = list(MANIFESTS) if args.analysis_set == "all" else [args.analysis_set]
    reports = [analyze_one(name) for name in selected]
    print(json.dumps({"status": "complete", "analyses": reports}, indent=2))


if __name__ == "__main__":
    main()
