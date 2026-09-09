"""Analyze the locked context-decomposition and raw-fraction LOTO additions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rat_kp_core.data import sha256_file
from rat_kp_core.physiology import PHYSIOLOGY_COLUMNS, load_physiology
from rat_kp_core.training import load_job_frames
from rat_kp_fusion.statistics import (
    benjamini_hochberg,
    exact_sign_flip_pvalue,
    loto_tissue_parent_cluster_ci,
    percentile_seed_ci,
    tissue_macro_rmse,
)
from rat_kp_fusion.training import metrics

from .config import load_config
from .jobs import build_plan
from .paths import ANALYSIS_DIR, CONFIG_PATH, EXPERIMENT_RESULT_DIR


BASE_ANALYSIS_DIR = (
    CONFIG_PATH.parents[1] / "injection_study" / "results" / "analysis"
)
CONTEXT_CONDITIONS = (
    "onehot_late", "onehot_early", "physiology_late", "physiology_early"
)


def load_predictions(plan: pd.DataFrame) -> pd.DataFrame:
    frames = []
    required = [
        "row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES",
        "Tissue", "y_true", "y_pred",
    ]
    for row in plan.itertuples(index=False):
        path = EXPERIMENT_RESULT_DIR / row.analysis / row.job_id / "predictions.csv"
        if not path.exists():
            raise RuntimeError(f"Missing additional-analysis predictions: {row.job_id}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if set(required) - set(frame.columns):
            raise RuntimeError(f"Invalid additional prediction schema: {row.job_id}")
        frame = frame[required].copy()
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite additional predictions: {row.job_id}")
        for column in (
            "job_id", "analysis", "stage", "model", "condition", "split_type",
            "split_seed", "heldout_tissue", "training_seed", "physiology_scaling",
        ):
            frame[column] = getattr(row, column)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def job_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "job_id", "analysis", "stage", "model", "condition", "split_type",
        "split_seed", "heldout_tissue", "training_seed", "physiology_scaling",
    ]
    rows = []
    for key, frame in predictions.groupby(grouping, sort=False):
        row = dict(zip(grouping, key))
        row.update(metrics(frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)))
        row["tissue_macro_rmse_log10"] = tissue_macro_rmse(frame)
        row["n_rows"] = len(frame)
        row["n_parent_groups"] = frame.parent_group_id.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def tissue_mean_baselines(plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    additive = plan[plan.analysis == "primary_context_decomposition"]
    rows = []
    for (split_type, split_seed), group in additive.groupby(["split_type", "split_seed"]):
        frames = load_job_frames(group.iloc[0])
        train = frames["train"]
        test = frames["test"]
        global_mean = float(train.log10Kp.mean())
        tissue_means = train.groupby("Tissue").log10Kp.mean().to_dict()
        if set(test.Tissue) - set(tissue_means):
            raise RuntimeError("Primary tissue-only baseline encountered an unseen tissue")
        predictions = {
            "training_global_mean": np.full(len(test), global_mean),
            "training_tissue_mean": test.Tissue.map(tissue_means).to_numpy(float),
        }
        for baseline, prediction in predictions.items():
            frame = pd.DataFrame({
                "Tissue": test.Tissue,
                "y_true": test.log10Kp.to_numpy(float),
                "y_pred": prediction,
            })
            row = {
                "split_type": split_type,
                "split_seed": int(split_seed),
                "baseline": baseline,
                **metrics(frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)),
                "tissue_macro_rmse_log10": tissue_macro_rmse(frame),
                "n_rows": len(frame),
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    wide = result.pivot(
        index=["split_type", "split_seed"], columns="baseline", values="rmse_log10"
    ).reset_index()
    summaries = []
    for split_type, frame in wide.groupby("split_type", sort=True):
        values = (
            frame["training_tissue_mean"] - frame["training_global_mean"]
        ).to_numpy(float)
        low, high = percentile_seed_ci(values)
        summaries.append({
            "split_type": split_type,
            "contrast": "training_tissue_mean_vs_training_global_mean",
            "n_split_seeds": len(values),
            "mean_delta_rmse": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "p_value_descriptive": exact_sign_flip_pvalue(values),
        })
    return result, pd.DataFrame(summaries)


def _paired_family(
    combined_metrics: pd.DataFrame,
    comparisons: dict[str, tuple[str, str]],
    family: str,
    expected_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = []
    for (model, split_type), frame in combined_metrics.groupby(["model", "split_type"]):
        wide = frame.pivot(index="split_seed", columns="condition", values="rmse_log10")
        for contrast, (comparison, reference) in comparisons.items():
            if comparison not in wide or reference not in wide:
                raise RuntimeError(f"Missing {family} contrast: {model}/{split_type}/{contrast}")
            for seed, value in (wide[comparison] - wide[reference]).items():
                deltas.append({
                    "model": model,
                    "split_type": split_type,
                    "family": family,
                    "contrast": contrast,
                    "comparison": comparison,
                    "reference": reference,
                    "split_seed": int(seed),
                    "delta_rmse": float(value),
                })
    delta_frame = pd.DataFrame(deltas)
    summaries = []
    grouping = ["model", "split_type", "family", "contrast", "comparison", "reference"]
    for key, frame in delta_frame.groupby(grouping, sort=True):
        values = frame.delta_rmse.to_numpy(float)
        low, high = percentile_seed_ci(values)
        row = dict(zip(grouping, key))
        row.update({
            "n_split_seeds": len(values),
            "mean_delta_rmse": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "p_value": exact_sign_flip_pvalue(values),
        })
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    if len(summary) != expected_size:
        raise RuntimeError(f"Unexpected {family} family size: {len(summary)}")
    summary["bh_q_value"] = benjamini_hochberg(summary.p_value.to_numpy(float))
    return delta_frame, summary


def primary_decomposition(new_metrics: pd.DataFrame):
    standard = pd.read_csv(
        BASE_ANALYSIS_DIR / "all_job_metrics.csv", encoding="utf-8-sig"
    )
    standard = standard[
        (standard.analysis_set == "primary") & (standard.stage == "part1")
    ].copy()
    additive = new_metrics[
        new_metrics.analysis == "primary_context_decomposition"
    ].copy()
    combined = pd.concat([
        standard[["model", "condition", "split_type", "split_seed", "rmse_log10"]],
        additive[["model", "condition", "split_type", "split_seed", "rmse_log10"]],
    ], ignore_index=True)
    additive_contrasts = {
        "additive_tissue_intercept_vs_structure": (
            "additive_tissue_intercept", "structure_only"
        )
    }
    context_contrasts = {
        f"{condition}_vs_additive_tissue_intercept": (
            condition, "additive_tissue_intercept"
        )
        for condition in CONTEXT_CONDITIONS
    }
    add_delta, add_summary = _paired_family(
        combined, additive_contrasts, "additive_vs_structure", 8
    )
    context_delta, context_summary = _paired_family(
        combined, context_contrasts, "context_vs_additive", 32
    )
    return add_delta, add_summary, context_delta, context_summary


def _average_loto(predictions: pd.DataFrame) -> pd.DataFrame:
    loto = predictions[predictions.analysis == "loto_raw_fraction_robustness"].copy()
    grouping = [
        "model", "condition", "Tissue", "parent_group_id", "identity_unit_id",
        "row_index", "Drug", "SMILES", "y_true",
    ]
    averaged = loto.groupby(grouping, as_index=False).agg(
        y_pred=("y_pred", "mean"),
        seed_sd=("y_pred", "std"),
        n_training_seeds=("training_seed", "nunique"),
    )
    if set(averaged.n_training_seeds) != {5}:
        raise RuntimeError("Raw-fraction LOTO requires five training seeds")
    return averaged


def _loto_position_analysis(averaged: pd.DataFrame):
    tissue_rows = []
    inference_rows = []
    for model, frame in averaged.groupby("model", sort=True):
        wide = frame.pivot(
            index=["Tissue", "parent_group_id", "row_index", "y_true"],
            columns="condition", values="y_pred",
        ).reset_index()
        deltas = []
        for tissue, tissue_frame in wide.groupby("Tissue", sort=True):
            truth = tissue_frame.y_true.to_numpy(float)
            early_rmse = float(np.sqrt(np.mean(
                (tissue_frame.physiology_early.to_numpy(float) - truth) ** 2
            )))
            late_rmse = float(np.sqrt(np.mean(
                (tissue_frame.physiology_late.to_numpy(float) - truth) ** 2
            )))
            delta = early_rmse - late_rmse
            deltas.append(delta)
            tissue_rows.append({
                "model": model,
                "Tissue": tissue,
                "early_rmse": early_rmse,
                "late_rmse": late_rmse,
                "early_minus_late_rmse": delta,
                "n_rows": len(tissue_frame),
                "n_parent_groups": tissue_frame.parent_group_id.nunique(),
            })
        low, high = loto_tissue_parent_cluster_ci(
            frame, "physiology_early", "physiology_late"
        )
        inference_rows.append({
            "model": model,
            "contrast": "physiology_early_raw_vs_physiology_late_raw",
            "n_tissues": len(deltas),
            "mean_tissue_equal_early_minus_late_rmse": float(np.mean(deltas)),
            "ci95_low": low,
            "ci95_high": high,
            "p_value": exact_sign_flip_pvalue(deltas),
            "positive_tissue_directions": int(np.sum(np.asarray(deltas) > 0)),
        })
    inference = pd.DataFrame(inference_rows)
    inference["bh_q_value"] = benjamini_hochberg(inference.p_value.to_numpy(float))
    return pd.DataFrame(tissue_rows), inference


def descriptor_extrapolation_audit() -> pd.DataFrame:
    matrix = load_physiology().set_index("tissue")[PHYSIOLOGY_COLUMNS]
    rows = []
    for heldout in matrix.index:
        train = matrix.drop(index=heldout).to_numpy(float)
        test = matrix.loc[heldout].to_numpy(float)
        mean = train.mean(axis=0)
        scale = train.std(axis=0, ddof=0)
        z_test = (test - mean) / scale
        z_train = (train - mean) / scale
        row = {
            "Tissue": heldout,
            "max_abs_train_only_z": float(np.max(np.abs(z_test))),
            "nearest_training_tissue_z_distance": float(np.min(np.linalg.norm(z_train - z_test, axis=1))),
        }
        row.update({
            f"z_{column}": float(value)
            for column, value in zip(PHYSIOLOGY_COLUMNS, z_test)
        })
        rows.append(row)
    return pd.DataFrame(rows)


def preprocessing_comparison(raw_tissues: pd.DataFrame, ood: pd.DataFrame):
    standard = pd.read_csv(
        BASE_ANALYSIS_DIR / "loto_tissue_effects.csv", encoding="utf-8-sig"
    )
    standard = standard[
        standard.contrast == "physiology_early_vs_late"
    ][["model", "Tissue", "comparison_rmse", "reference_rmse", "delta_rmse"]].rename(columns={
        "comparison_rmse": "standardized_early_rmse",
        "reference_rmse": "standardized_late_rmse",
        "delta_rmse": "standardized_early_minus_late_rmse",
    })
    comparison = standard.merge(raw_tissues, on=["model", "Tissue"], validate="one_to_one")
    comparison = comparison.merge(ood, on="Tissue", validate="many_to_one")
    comparison["raw_minus_standardized_early_rmse"] = (
        comparison.early_rmse - comparison.standardized_early_rmse
    )
    comparison["raw_minus_standardized_late_rmse"] = (
        comparison.late_rmse - comparison.standardized_late_rmse
    )
    associations = []
    for model, frame in comparison.groupby("model", sort=True):
        for distance in ("max_abs_train_only_z", "nearest_training_tissue_z_distance"):
            associations.append({
                "model": model,
                "distance": distance,
                "n_tissues": len(frame),
                "spearman_standardized_gap": float(frame[distance].corr(
                    frame.standardized_early_minus_late_rmse, method="spearman"
                )),
                "spearman_raw_gap": float(frame[distance].corr(
                    frame.early_minus_late_rmse, method="spearman"
                )),
            })
    return comparison, pd.DataFrame(associations)


def main() -> None:
    load_config(require_locked=True)
    plan = build_plan()
    predictions = load_predictions(plan)
    metric_frame = job_metrics(predictions)
    tissue_baselines, tissue_baseline_summary = tissue_mean_baselines(plan)
    add_delta, add_summary, context_delta, context_summary = primary_decomposition(metric_frame)
    raw_averaged = _average_loto(predictions)
    raw_tissues, raw_inference = _loto_position_analysis(raw_averaged)
    ood = descriptor_extrapolation_audit()
    preprocessing, associations = preprocessing_comparison(raw_tissues, ood)
    outputs = {
        "all_additional_job_metrics.csv": metric_frame,
        "tissue_only_baseline_metrics.csv": tissue_baselines,
        "tissue_only_vs_global_summary.csv": tissue_baseline_summary,
        "additive_vs_structure_seed_deltas.csv": add_delta,
        "additive_vs_structure_effects.csv": add_summary,
        "context_vs_additive_seed_deltas.csv": context_delta,
        "context_vs_additive_effects.csv": context_summary,
        "loto_raw_seed_averaged_predictions.csv": raw_averaged,
        "loto_raw_tissue_effects.csv": raw_tissues,
        "loto_raw_inference.csv": raw_inference,
        "descriptor_extrapolation_audit.csv": ood,
        "loto_preprocessing_comparison.csv": preprocessing,
        "loto_ood_gap_associations.csv": associations,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(ANALYSIS_DIR / filename, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete",
        "additional_config_sha256": sha256_file(CONFIG_PATH),
        "planned_jobs": len(plan),
        "prediction_rows": len(predictions),
        "outputs": {
            name: {"rows": len(frame), "sha256": sha256_file(ANALYSIS_DIR / name)}
            for name, frame in outputs.items()
        },
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "outputs": list(outputs)}, indent=2))

