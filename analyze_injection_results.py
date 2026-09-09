"""Analyze the complete injection study using only prespecified paired contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from rat_kp_core.data import sha256_file

from rat_kp_fusion.config import load_config
from rat_kp_fusion.jobs import build_plan
from rat_kp_fusion.paths import CONFIG_PATH, EXPERIMENT_RESULT_DIR, RESULT_DIR, STUDY_ROOT
from rat_kp_fusion.statistics import (
    benjamini_hochberg,
    exact_sign_flip_pvalue,
    loto_tissue_parent_cluster_ci,
    percentile_seed_ci,
    tissue_macro_rmse,
)
from rat_kp_fusion.training import metrics


ANALYSIS_DIR = RESULT_DIR / "analysis"
CONTEXT_CONTRASTS = {
    "onehot_late_vs_structure": ("onehot_late", "structure_only"),
    "onehot_early_vs_structure": ("onehot_early", "structure_only"),
    "physiology_late_vs_structure": ("physiology_late", "structure_only"),
    "physiology_early_vs_structure": ("physiology_early", "structure_only"),
}
POSITION_CONTRASTS = {
    "onehot_early_vs_late": ("onehot_early", "onehot_late"),
    "physiology_early_vs_late": ("physiology_early", "physiology_late"),
}
LOTO_CONTRASTS = {
    "physiology_late_vs_structure": ("physiology_late", "structure_only"),
    "physiology_early_vs_structure": ("physiology_early", "structure_only"),
    "physiology_early_vs_late": ("physiology_early", "physiology_late"),
}


def _prediction_path(row) -> Path:
    if row.execution_mode == "legacy_reuse":
        return STUDY_ROOT / row.legacy_result_path / "predictions.csv"
    return (
        EXPERIMENT_RESULT_DIR / row.analysis_set / row.stage / row.job_id / "predictions.csv"
    )


def load_complete_predictions(plan: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in plan.itertuples(index=False):
        path = _prediction_path(row)
        if not path.exists():
            raise RuntimeError(f"Missing planned predictions: {row.job_id}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {
            "row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES",
            "Tissue", "y_true", "y_pred",
        }
        if required - set(frame.columns) or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all():
            raise RuntimeError(f"Invalid prediction file: {row.job_id}")
        frame = frame[list(required)].copy()
        for column in (
            "job_id", "stage", "analysis_set", "model", "condition", "split_type",
            "split_seed", "heldout_tissue", "training_seed",
        ):
            frame[column] = getattr(row, column)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def job_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "job_id", "stage", "analysis_set", "model", "condition", "split_type",
        "split_seed", "heldout_tissue", "training_seed",
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


def _part1_family(metrics_frame: pd.DataFrame, contrasts: dict, family: str):
    part1 = metrics_frame[metrics_frame.stage == "part1"]
    deltas = []
    for (analysis_set, model, split_type), frame in part1.groupby(
        ["analysis_set", "model", "split_type"], sort=True
    ):
        wide = frame.pivot(index="split_seed", columns="condition", values="rmse_log10")
        for name, (comparison, reference) in contrasts.items():
            if comparison not in wide or reference not in wide:
                raise RuntimeError(f"Missing Part 1 contrast: {analysis_set}/{model}/{split_type}/{name}")
            for seed, value in (wide[comparison] - wide[reference]).items():
                deltas.append({
                    "analysis_set": analysis_set,
                    "model": model,
                    "split_type": split_type,
                    "family": family,
                    "contrast": name,
                    "comparison": comparison,
                    "reference": reference,
                    "split_seed": int(seed),
                    "delta_rmse": float(value),
                })
    delta_frame = pd.DataFrame(deltas)
    summaries = []
    grouping = ["analysis_set", "model", "split_type", "family", "contrast", "comparison", "reference"]
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
    summary["bh_q_value"] = np.nan
    primary = summary.analysis_set == "primary"
    if primary.sum() != (32 if family == "context_vs_structure" else 16):
        raise RuntimeError(f"Unexpected primary {family} family size")
    summary.loc[primary, "bh_q_value"] = benjamini_hochberg(
        summary.loc[primary, "p_value"].to_numpy(float)
    )
    return delta_frame, summary


def loto_analysis(predictions: pd.DataFrame):
    loto = predictions[predictions.stage == "loto"].copy()
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
        raise RuntimeError("LOTO requires five training seeds per prediction")
    tissue_rows = []
    inference = []
    for model, model_frame in averaged.groupby("model", sort=True):
        wide = model_frame.pivot(
            index=["Tissue", "parent_group_id", "row_index", "y_true"],
            columns="condition",
            values="y_pred",
        ).reset_index()
        for name, (comparison, reference) in LOTO_CONTRASTS.items():
            deltas = []
            for tissue, tissue_frame in wide.groupby("Tissue", sort=True):
                truth = tissue_frame.y_true.to_numpy(float)
                comp_rmse = np.sqrt(np.mean((tissue_frame[comparison].to_numpy(float) - truth) ** 2))
                ref_rmse = np.sqrt(np.mean((tissue_frame[reference].to_numpy(float) - truth) ** 2))
                delta = float(comp_rmse - ref_rmse)
                deltas.append(delta)
                tissue_rows.append({
                    "model": model, "contrast": name, "Tissue": tissue,
                    "comparison_rmse": comp_rmse, "reference_rmse": ref_rmse,
                    "delta_rmse": delta, "n_rows": len(tissue_frame),
                    "n_parent_groups": tissue_frame.parent_group_id.nunique(),
                })
            low, high = loto_tissue_parent_cluster_ci(model_frame, comparison, reference)
            inference.append({
                "model": model,
                "contrast": name,
                "comparison": comparison,
                "reference": reference,
                "n_tissues": len(deltas),
                "mean_tissue_equal_delta_rmse": float(np.mean(deltas)),
                "ci95_low": low,
                "ci95_high": high,
                "p_value": exact_sign_flip_pvalue(deltas),
            })
    inference = pd.DataFrame(inference)
    inference["secondary_bh_q_value"] = benjamini_hochberg(inference.p_value.to_numpy(float))
    return averaged, pd.DataFrame(tissue_rows), inference


def _descriptor_table(predictions: pd.DataFrame) -> pd.DataFrame:
    config = load_config()["property_stratification"]
    unique = predictions[["SMILES"]].drop_duplicates().copy()
    functions = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "RotB": Descriptors.NumRotatableBonds,
    }
    for name, function in functions.items():
        unique[name] = unique.SMILES.map(lambda value: float(function(Chem.MolFromSmiles(value))))
        low, high = config[name]
        unique[f"{name}_bin"] = pd.cut(
            unique[name], [-np.inf, low, high, np.inf], labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype(str)
    return unique


def property_stratification(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[(predictions.analysis_set == "primary") & (predictions.stage == "part1")].copy()
    primary = primary.merge(_descriptor_table(primary), on="SMILES", validate="many_to_one")
    rows = []
    for descriptor in ("MolWt", "MolLogP", "TPSA", "RotB"):
        bin_column = f"{descriptor}_bin"
        grouping = ["model", "condition", "split_type", "split_seed", bin_column]
        for key, frame in primary.groupby(grouping, sort=True):
            values = metrics(frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float))
            rows.append({
                "descriptor": descriptor,
                "bin": key[-1],
                "model": key[0],
                "condition": key[1],
                "split_type": key[2],
                "split_seed": key[3],
                "n_rows": len(frame),
                **values,
            })
    return pd.DataFrame(rows)


def main() -> None:
    load_config(require_frozen=True)
    plan = build_plan()
    predictions = load_complete_predictions(plan)
    metrics_frame = job_metrics(predictions)
    context_deltas, context_summary = _part1_family(
        metrics_frame, CONTEXT_CONTRASTS, "context_vs_structure"
    )
    position_deltas, position_summary = _part1_family(
        metrics_frame, POSITION_CONTRASTS, "early_vs_late"
    )
    loto_predictions, loto_tissues, loto_inference = loto_analysis(predictions)
    property_results = property_stratification(predictions)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_job_metrics.csv": metrics_frame,
        "part1_context_seed_deltas.csv": context_deltas,
        "part1_context_effects.csv": context_summary,
        "part1_position_seed_deltas.csv": position_deltas,
        "part1_position_effects.csv": position_summary,
        "loto_seed_averaged_predictions.csv": loto_predictions,
        "loto_tissue_effects.csv": loto_tissues,
        "loto_inference.csv": loto_inference,
        "property_stratified_metrics.csv": property_results,
    }
    for filename, frame in outputs.items():
        frame.to_csv(ANALYSIS_DIR / filename, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete",
        "config_sha256": sha256_file(CONFIG_PATH),
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


if __name__ == "__main__":
    main()
