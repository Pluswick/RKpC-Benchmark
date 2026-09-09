"""Exploratory deep-dive analysis of the completed injection study.

Confirmatory results remain those defined in analyze_injection_results.py.  The
additional comparisons produced here are explicitly labelled exploratory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_injection_results import ANALYSIS_DIR, load_complete_predictions
from rat_kp_fusion.jobs import build_plan
from rat_kp_fusion.statistics import (
    benjamini_hochberg,
    exact_sign_flip_pvalue,
    loto_tissue_parent_cluster_ci,
    percentile_seed_ci,
)
from rat_kp_core.physiology import ContextEncoder, load_physiology
from rat_kp_core.training import load_job_frames


OUT_DIR = ANALYSIS_DIR / "deep_dive"
MODEL_ORDER = ["gcn", "gine", "d_mpnn", "attentive_fp"]
CONTEXT_CONTRASTS = {
    "onehot_late_vs_structure": ("onehot_late", "structure_only"),
    "onehot_early_vs_structure": ("onehot_early", "structure_only"),
    "physiology_late_vs_structure": ("physiology_late", "structure_only"),
    "physiology_early_vs_structure": ("physiology_early", "structure_only"),
}
LOTO_CONTRASTS = {
    "physiology_late_vs_structure": ("physiology_late", "structure_only"),
    "physiology_early_vs_structure": ("physiology_early", "structure_only"),
    "physiology_early_vs_late": ("physiology_early", "physiology_late"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(float)
    low, high = percentile_seed_ci(array)
    return {
        "n_split_seeds": len(array),
        "mean_delta_rmse": float(array.mean()),
        "sd_delta_rmse": float(array.std(ddof=1)),
        "ci95_low": low,
        "ci95_high": high,
        "p_value": exact_sign_flip_pvalue(array),
    }


def primary_performance(job_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = job_metrics[
        (job_metrics.analysis_set == "primary") & (job_metrics.stage == "part1")
    ]
    grouping = ["split_type", "model", "condition"]
    return primary.groupby(grouping, as_index=False).agg(
        n_split_seeds=("split_seed", "nunique"),
        mean_rmse=("rmse_log10", "mean"),
        sd_rmse=("rmse_log10", "std"),
        min_rmse=("rmse_log10", "min"),
        max_rmse=("rmse_log10", "max"),
        mean_mae=("mae_log10", "mean"),
        mean_within_two_fold=("within_two_fold", "mean"),
        mean_within_three_fold=("within_three_fold", "mean"),
        mean_tissue_macro_rmse=("tissue_macro_rmse_log10", "mean"),
    )


def representation_effects(job_metrics: pd.DataFrame) -> pd.DataFrame:
    """Exploratory one-hot minus physiology paired effects."""
    primary = job_metrics[
        (job_metrics.analysis_set == "primary") & (job_metrics.stage == "part1")
    ]
    wide = primary.pivot(
        index=["model", "split_type", "split_seed"],
        columns="condition",
        values="rmse_log10",
    ).reset_index()
    rows = []
    for position in ("late", "early"):
        comparison = f"onehot_{position}"
        reference = f"physiology_{position}"
        wide[f"delta_{position}"] = wide[comparison] - wide[reference]
        for (model, split_type), frame in wide.groupby(["model", "split_type"], sort=True):
            row = {
                "model": model,
                "split_type": split_type,
                "contrast": f"onehot_vs_physiology_{position}",
                "comparison": comparison,
                "reference": reference,
            }
            row.update(_paired_summary(frame[f"delta_{position}"]))
            rows.append(row)
    result = pd.DataFrame(rows)
    result["exploratory_bh_q_value"] = benjamini_hochberg(result.p_value.to_numpy(float))
    return result


def model_pair_effects(job_metrics: pd.DataFrame) -> pd.DataFrame:
    """Exploratory paired model comparisons within identical split/condition."""
    primary = job_metrics[
        (job_metrics.analysis_set == "primary") & (job_metrics.stage == "part1")
    ]
    wide = primary.pivot(
        index=["split_type", "condition", "split_seed"],
        columns="model",
        values="rmse_log10",
    ).reset_index()
    rows = []
    for comparison, reference in itertools.combinations(MODEL_ORDER, 2):
        delta_column = f"{comparison}_minus_{reference}"
        wide[delta_column] = wide[comparison] - wide[reference]
        for (split_type, condition), frame in wide.groupby(
            ["split_type", "condition"], sort=True
        ):
            row = {
                "split_type": split_type,
                "condition": condition,
                "comparison": comparison,
                "reference": reference,
            }
            row.update(_paired_summary(frame[delta_column]))
            rows.append(row)
    result = pd.DataFrame(rows)
    result["exploratory_bh_q_value"] = benjamini_hochberg(result.p_value.to_numpy(float))
    return result


def primary_tissue_results(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = predictions[
        (predictions.analysis_set == "primary") & (predictions.stage == "part1")
    ].copy()
    primary["residual"] = primary.y_pred - primary.y_true
    primary["absolute_error"] = primary.residual.abs()
    primary["squared_error"] = primary.residual ** 2
    grouping = ["model", "condition", "split_type", "split_seed", "Tissue"]
    per_seed = primary.groupby(grouping, as_index=False).agg(
        n_rows=("row_index", "size"),
        n_parent_groups=("parent_group_id", "nunique"),
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
        bias=("residual", "mean"),
    )
    per_seed["rmse"] = np.sqrt(per_seed.pop("mse"))
    summary = per_seed.groupby(
        ["model", "condition", "split_type", "Tissue"], as_index=False
    ).agg(
        n_observed_seeds=("split_seed", "nunique"),
        mean_test_rows=("n_rows", "mean"),
        mean_rmse=("rmse", "mean"),
        sd_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_bias=("bias", "mean"),
    )

    wide = per_seed.pivot(
        index=["model", "split_type", "split_seed", "Tissue"],
        columns="condition",
        values="rmse",
    ).reset_index()
    rows = []
    for name, (comparison, reference) in CONTEXT_CONTRASTS.items():
        delta = wide[comparison] - wide[reference]
        block = wide[["model", "split_type", "split_seed", "Tissue"]].copy()
        block["contrast"] = name
        block["delta_rmse"] = delta
        rows.append(block)
    deltas = pd.concat(rows, ignore_index=True)
    delta_summary = deltas.groupby(
        ["model", "split_type", "Tissue", "contrast"], as_index=False
    ).agg(
        n_observed_seeds=("split_seed", "nunique"),
        mean_delta_rmse=("delta_rmse", "mean"),
        sd_delta_rmse=("delta_rmse", "std"),
        seeds_improved=("delta_rmse", lambda values: int((values < 0).sum())),
    )
    return summary, delta_summary


def loto_tissue_performance(loto_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = loto_predictions.copy()
    frame["residual"] = frame.y_pred - frame.y_true
    frame["absolute_error"] = frame.residual.abs()
    frame["squared_error"] = frame.residual ** 2
    result = frame.groupby(["model", "condition", "Tissue"], as_index=False).agg(
        n_rows=("row_index", "size"),
        n_parent_groups=("parent_group_id", "nunique"),
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
        bias=("residual", "mean"),
        mean_seed_sd=("seed_sd", "mean"),
    )
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result


def loto_cross_model_summary(loto_tissue_effects: pd.DataFrame) -> pd.DataFrame:
    return loto_tissue_effects.groupby(["contrast", "Tissue"], as_index=False).agg(
        n_models=("model", "nunique"),
        mean_delta_rmse=("delta_rmse", "mean"),
        min_delta_rmse=("delta_rmse", "min"),
        max_delta_rmse=("delta_rmse", "max"),
        models_comparison_better=("delta_rmse", lambda values: int((values < 0).sum())),
        models_comparison_worse=("delta_rmse", lambda values: int((values > 0).sum())),
    )


def loto_leave_one_tissue_influence(loto_tissue_effects: pd.DataFrame) -> pd.DataFrame:
    """Systematic post-hoc influence analysis excluding every tissue in turn."""
    rows = []
    for excluded_tissue in sorted(loto_tissue_effects.Tissue.unique()):
        filtered = loto_tissue_effects[
            loto_tissue_effects.Tissue != excluded_tissue
        ]
        for (model, contrast), frame in filtered.groupby(
            ["model", "contrast"], sort=True
        ):
            values = frame.delta_rmse.to_numpy(float)
            rows.append({
                "classification": "post_hoc_exploratory_influence_analysis",
                "excluded_tissue": excluded_tissue,
                "model": model,
                "contrast": contrast,
                "n_tissues": len(values),
                "mean_delta_rmse": float(values.mean()),
                "p_value": exact_sign_flip_pvalue(values),
            })
    result = pd.DataFrame(rows)
    result["bh_q_within_exclusion"] = np.nan
    for _, frame in result.groupby("excluded_tissue", sort=True):
        result.loc[frame.index, "bh_q_within_exclusion"] = benjamini_hochberg(
            frame.p_value.to_numpy(float)
        )
    return result


def loto_excluding_adipose(loto_predictions: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc sensitivity prompted by the extreme adipose physiology vector."""
    filtered = loto_predictions[loto_predictions.Tissue != "adipose"].copy()
    rows = []
    for model, model_frame in filtered.groupby("model", sort=True):
        wide = model_frame.pivot(
            index=["Tissue", "parent_group_id", "row_index", "y_true"],
            columns="condition",
            values="y_pred",
        ).reset_index()
        for name, (comparison, reference) in LOTO_CONTRASTS.items():
            tissue_deltas = []
            for _, tissue_frame in wide.groupby("Tissue", sort=True):
                truth = tissue_frame.y_true.to_numpy(float)
                comparison_rmse = np.sqrt(
                    np.mean((tissue_frame[comparison].to_numpy(float) - truth) ** 2)
                )
                reference_rmse = np.sqrt(
                    np.mean((tissue_frame[reference].to_numpy(float) - truth) ** 2)
                )
                tissue_deltas.append(comparison_rmse - reference_rmse)
            low, high = loto_tissue_parent_cluster_ci(model_frame, comparison, reference)
            rows.append({
                "classification": "post_hoc_exploratory_sensitivity",
                "excluded_tissue": "adipose",
                "model": model,
                "contrast": name,
                "comparison": comparison,
                "reference": reference,
                "n_tissues": len(tissue_deltas),
                "mean_tissue_equal_delta_rmse": float(np.mean(tissue_deltas)),
                "ci95_low": low,
                "ci95_high": high,
                "p_value": exact_sign_flip_pvalue(tissue_deltas),
            })
    result = pd.DataFrame(rows)
    result["exploratory_bh_q_value"] = benjamini_hochberg(result.p_value.to_numpy(float))
    return result


def loto_context_extrapolation(plan: pd.DataFrame) -> pd.DataFrame:
    physiology = load_physiology()
    feature_names = list(physiology.select_dtypes(include=[np.number]).columns)
    loto = plan[
        (plan.stage == "loto") & (plan.condition == "physiology_late")
    ]
    rows = []
    for tissue in sorted(loto.heldout_tissue.unique()):
        job = loto[loto.heldout_tissue == tissue].iloc[0]
        frames = load_job_frames(job)
        encoder = ContextEncoder(
            "physiology", physiology, scaling=str(job.physiology_scaling)
        ).fit(frames["train"])
        vector = encoder.transform(frames["test"])[0].astype(float)
        row = {
            "Tissue": tissue,
            "n_test_rows": len(frames["test"]),
            "max_abs_train_z": float(np.abs(vector).max()),
            "l2_train_z": float(np.linalg.norm(vector)),
        }
        row.update({f"train_z__{name}": value for name, value in zip(feature_names, vector)})
        rows.append(row)
    return pd.DataFrame(rows)


def property_context_summary(property_metrics: pd.DataFrame) -> pd.DataFrame:
    index = ["descriptor", "bin", "model", "split_type", "split_seed"]
    wide = property_metrics.pivot(index=index, columns="condition", values="rmse_log10").reset_index()
    rows = []
    for name, (comparison, reference) in CONTEXT_CONTRASTS.items():
        block = wide[index].copy()
        block["contrast"] = name
        block["delta_rmse"] = wide[comparison] - wide[reference]
        rows.append(block)
    deltas = pd.concat(rows, ignore_index=True)
    return deltas.groupby(
        ["descriptor", "bin", "model", "split_type", "contrast"], as_index=False
    ).agg(
        n_split_seeds=("split_seed", "nunique"),
        mean_delta_rmse=("delta_rmse", "mean"),
        sd_delta_rmse=("delta_rmse", "std"),
        seeds_improved=("delta_rmse", lambda values: int((values < 0).sum())),
    )


def sensitivity_direction_summary(
    context_effects: pd.DataFrame, position_effects: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for family, frame in (
        ("context_vs_structure", context_effects),
        ("early_vs_late", position_effects),
    ):
        index = ["model", "split_type", "contrast"]
        wide = frame.pivot(index=index, columns="analysis_set", values="mean_delta_rmse").reset_index()
        wide["family"] = family
        wide["all_three_negative"] = (
            (wide.primary < 0) & (wide.ad_excluded < 0) & (wide.condition_specific < 0)
        )
        wide["ad_same_direction_as_primary"] = wide.primary * wide.ad_excluded > 0
        wide["condition_same_direction_as_primary"] = (
            wide.primary * wide.condition_specific > 0
        )
        rows.append(wide)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    plan = build_plan()
    predictions = load_complete_predictions(plan)
    job_metrics = pd.read_csv(ANALYSIS_DIR / "all_job_metrics.csv", encoding="utf-8-sig")
    context_effects = pd.read_csv(
        ANALYSIS_DIR / "part1_context_effects.csv", encoding="utf-8-sig"
    )
    position_effects = pd.read_csv(
        ANALYSIS_DIR / "part1_position_effects.csv", encoding="utf-8-sig"
    )
    loto_predictions = pd.read_csv(
        ANALYSIS_DIR / "loto_seed_averaged_predictions.csv", encoding="utf-8-sig"
    )
    loto_tissue_effects = pd.read_csv(
        ANALYSIS_DIR / "loto_tissue_effects.csv", encoding="utf-8-sig"
    )
    property_metrics = pd.read_csv(
        ANALYSIS_DIR / "property_stratified_metrics.csv", encoding="utf-8-sig"
    )

    tissue_performance, tissue_context = primary_tissue_results(predictions)
    outputs = {
        "primary_performance_summary.csv": primary_performance(job_metrics),
        "exploratory_representation_effects.csv": representation_effects(job_metrics),
        "exploratory_model_pair_effects.csv": model_pair_effects(job_metrics),
        "primary_tissue_performance.csv": tissue_performance,
        "primary_tissue_context_effects.csv": tissue_context,
        "loto_tissue_performance.csv": loto_tissue_performance(loto_predictions),
        "loto_cross_model_tissue_summary.csv": loto_cross_model_summary(loto_tissue_effects),
        "loto_leave_one_tissue_influence.csv": loto_leave_one_tissue_influence(
            loto_tissue_effects
        ),
        "loto_excluding_adipose_sensitivity.csv": loto_excluding_adipose(loto_predictions),
        "loto_context_extrapolation.csv": loto_context_extrapolation(plan),
        "property_context_summary.csv": property_context_summary(property_metrics),
        "sensitivity_direction_summary.csv": sensitivity_direction_summary(
            context_effects, position_effects
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete",
        "classification": "exploratory_deep_dive",
        "planned_jobs": len(plan),
        "prediction_rows_loaded": len(predictions),
        "outputs": {},
    }
    for name, frame in outputs.items():
        path = OUT_DIR / name
        manifest["outputs"][name] = {"rows": len(frame), "sha256": _sha256(path)}
    (OUT_DIR / "deep_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
