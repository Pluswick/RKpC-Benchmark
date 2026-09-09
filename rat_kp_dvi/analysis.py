from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rat_kp_fusion.jobs import build_plan as build_injection_plan
from rat_kp_fusion.paths import EXPERIMENT_RESULT_DIR as INJECTION_DIR, STUDY_ROOT
from rat_kp_controls.jobs import build_plan as build_additional_plan
from rat_kp_controls.paths import EXPERIMENT_RESULT_DIR as ADDITIONAL_DIR
from .config import load_config
from .paths import ANALYSIS_DIR, EXPERIMENT_DIR


def _injection_prediction_path(row: pd.Series) -> Path:
    if str(row["execution_mode"]) == "legacy_reuse":
        return STUDY_ROOT / str(row["legacy_result_path"]) / "predictions.csv"
    return INJECTION_DIR / str(row["analysis_set"]) / str(row["stage"]) / str(row["job_id"]) / "predictions.csv"


def _job(split_type: str, seed: int, model: str, condition: str) -> pd.Series:
    plan = build_injection_plan()
    rows = plan[
        plan["analysis_set"].eq("primary") & plan["stage"].eq("part1")
        & plan["split_type"].eq(split_type) & plan["split_seed"].eq(seed)
        & plan["model"].eq(model) & plan["condition"].eq(condition)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Injection job lookup failed: {split_type}/{seed}/{model}/{condition}")
    return rows.iloc[0]


def _additive_job(split_type: str, seed: int, model: str) -> pd.Series:
    plan = build_additional_plan()
    rows = plan[
        plan["analysis"].eq("primary_context_decomposition")
        & plan["split_type"].eq(split_type) & plan["split_seed"].eq(seed)
        & plan["model"].eq(model) & plan["condition"].eq("additive_tissue_intercept")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Additive job lookup failed: {split_type}/{seed}/{model}")
    return rows.iloc[0]


def _read_primary_predictions(split_type: str, seed: int, model: str, full_condition: str) -> dict[str, pd.DataFrame]:
    structure = pd.read_csv(_injection_prediction_path(_job(split_type, seed, model, "structure_only")), encoding="utf-8-sig")
    full = pd.read_csv(_injection_prediction_path(_job(split_type, seed, model, full_condition)), encoding="utf-8-sig")
    additive_row = _additive_job(split_type, seed, model)
    additive = pd.read_csv(ADDITIONAL_DIR / additive_row.analysis / additive_row.job_id / "predictions.csv", encoding="utf-8-sig")
    expected = set(structure["row_index"])
    if set(full["row_index"]) != expected or set(additive["row_index"]) != expected:
        raise RuntimeError("Primary comparator panels are not row matched")
    return {"structure_only": structure, "additive_tissue_intercept": additive, "full_context": full}


def _aggregate_panel(frame: pd.DataFrame, volumes: dict[str, float], bw: float) -> pd.DataFrame:
    work = frame.copy()
    work["volume_ml"] = work["Tissue"].map(volumes)
    if work["volume_ml"].isna().any():
        raise RuntimeError(f"Unmapped tissues: {sorted(work.loc[work.volume_ml.isna(), 'Tissue'].unique())}")
    work["true_component"] = work["volume_ml"] * np.power(10.0, work["y_true"].astype(float)) / bw
    work["pred_component"] = work["volume_ml"] * np.power(10.0, work["y_pred"].astype(float)) / bw
    keys = ["parent_group_id", "identity_unit_id", "Drug", "SMILES"]
    out = work.groupby(keys, as_index=False).agg(
        n_tissues=("Tissue", "nunique"),
        tissues=("Tissue", lambda values: "|".join(sorted(set(values)))),
        observed_dvi_true_L_per_kg=("true_component", "sum"),
        observed_dvi_pred_L_per_kg=("pred_component", "sum"),
    )
    out["y_true_log10_dvi"] = np.log10(out["observed_dvi_true_L_per_kg"])
    out["y_pred_log10_dvi"] = np.log10(out["observed_dvi_pred_L_per_kg"])
    out["residual_log10"] = out["y_pred_log10_dvi"] - out["y_true_log10_dvi"]
    out["absolute_fold_error"] = np.power(10.0, np.abs(out["residual_log10"]))
    return out


def build_observed_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(); volumes = cfg["tissue_volumes_ml"]; bw = float(cfg["body_weight_g"])
    rows = []
    for split_type, spec in cfg["representative_models"].items():
        for seed in cfg["split_seeds"]:
            frames = _read_primary_predictions(split_type, seed, spec["model"], spec["full_context"])
            for method, frame in frames.items():
                aggregated = _aggregate_panel(frame, volumes, bw)
                aggregated.insert(0, "method", method)
                aggregated.insert(0, "model", spec["model"])
                aggregated.insert(0, "split_seed", seed)
                aggregated.insert(0, "split_type", split_type)
                rows.append(aggregated)
    seed_identity = pd.concat(rows, ignore_index=True)
    keys = ["split_type", "model", "method", "parent_group_id", "identity_unit_id", "Drug", "SMILES"]
    averaged = seed_identity.groupby(keys, as_index=False).agg(
        n_oof_seeds=("split_seed", "nunique"), n_tissues=("n_tissues", "first"), tissues=("tissues", "first"),
        observed_dvi_true_L_per_kg=("observed_dvi_true_L_per_kg", "first"),
        observed_dvi_pred_L_per_kg=("observed_dvi_pred_L_per_kg", "mean"),
    )
    averaged["y_true_log10_dvi"] = np.log10(averaged["observed_dvi_true_L_per_kg"])
    averaged["y_pred_log10_dvi"] = np.log10(averaged["observed_dvi_pred_L_per_kg"])
    averaged["residual_log10"] = averaged["y_pred_log10_dvi"] - averaged["y_true_log10_dvi"]
    averaged["absolute_fold_error"] = 10 ** averaged["residual_log10"].abs()
    return seed_identity, averaged


def _two_way_full_grid(frame: pd.DataFrame, tissues: list[str]) -> tuple[pd.DataFrame, float]:
    work = frame.copy()
    identities = sorted(work["identity_unit_id"].unique().tolist(), key=str)
    observed_tissues = sorted(work["Tissue"].unique().tolist())
    if set(observed_tissues) != set(tissues):
        raise RuntimeError("Cannot reconstruct additive full grid because a tissue is absent from the test panel")
    identity_index = {value: idx for idx, value in enumerate(identities)}
    tissue_index = {value: idx for idx, value in enumerate(tissues)}
    x = np.zeros((len(work), len(identities) + len(tissues) - 1), dtype=float)
    for row_idx, row in enumerate(work.itertuples(index=False)):
        x[row_idx, identity_index[row.identity_unit_id]] = 1.0
        t_idx = tissue_index[row.Tissue]
        if t_idx > 0:
            x[row_idx, len(identities) + t_idx - 1] = 1.0
    coef, *_ = np.linalg.lstsq(x, work["y_pred"].to_numpy(float), rcond=None)
    fitted = x @ coef
    max_residual = float(np.max(np.abs(fitted - work["y_pred"].to_numpy(float))))
    meta = work[["identity_unit_id", "parent_group_id", "Drug", "SMILES"]].drop_duplicates("identity_unit_id").set_index("identity_unit_id")
    rows = []
    for identity in identities:
        for tissue in tissues:
            pred = coef[identity_index[identity]]
            t_idx = tissue_index[tissue]
            if t_idx > 0:
                pred += coef[len(identities) + t_idx - 1]
            values = meta.loc[identity]
            rows.append({"identity_unit_id": identity, "parent_group_id": values.parent_group_id, "Drug": values.Drug, "SMILES": values.SMILES, "Tissue": tissue, "y_pred": pred})
    return pd.DataFrame(rows), max_residual


def build_full_grid() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(); tissues = list(cfg["tissue_volumes_ml"]); volumes = cfg["tissue_volumes_ml"]; bw = float(cfg["body_weight_g"])
    split_type = "parent_group"; spec = cfg["representative_models"][split_type]
    grid_rows = []; dvi_rows = []; recon_rows = []
    for seed in cfg["split_seeds"]:
        canonical = _read_primary_predictions(split_type, seed, spec["model"], spec["full_context"])
        full_job = _job(split_type, seed, spec["model"], spec["full_context"])
        full = pd.read_csv(EXPERIMENT_DIR / split_type / full_job.job_id / "full_grid_predictions.csv", encoding="utf-8-sig")
        structure = canonical["structure_only"].groupby(["identity_unit_id", "parent_group_id", "Drug", "SMILES"], as_index=False)["y_pred"].mean()
        structure = structure.merge(pd.DataFrame({"Tissue": tissues}), how="cross")
        additive, residual = _two_way_full_grid(canonical["additive_tissue_intercept"], tissues)
        recon_rows.append({"split_seed": seed, "additive_max_reconstruction_residual_log10": residual})
        for method, frame in {"structure_only": structure, "additive_tissue_intercept": additive, "full_context": full}.items():
            x = frame.copy(); x["method"] = method; x["split_seed"] = seed
            x["volume_ml"] = x["Tissue"].map(volumes)
            x["kp_pred"] = 10 ** x["y_pred"].astype(float)
            x["dvi_component_L_per_kg"] = x["volume_ml"] * x["kp_pred"] / bw
            grid_rows.append(x[["split_seed", "method", "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue", "volume_ml", "y_pred", "kp_pred", "dvi_component_L_per_kg"]])
            agg = x.groupby(["parent_group_id", "identity_unit_id", "Drug", "SMILES"], as_index=False)["dvi_component_L_per_kg"].sum()
            agg = agg.rename(columns={"dvi_component_L_per_kg": "dvi_11_pred_L_per_kg"})
            agg["partial_vss_11_Rb1_L_per_kg"] = agg["dvi_11_pred_L_per_kg"] + float(cfg["blood_volume_ml"]) / bw
            agg["split_seed"] = seed; agg["method"] = method
            dvi_rows.append(agg)
    grid = pd.concat(grid_rows, ignore_index=True)
    dvi = pd.concat(dvi_rows, ignore_index=True)
    dvi_avg = dvi.groupby(["method", "parent_group_id", "identity_unit_id", "Drug", "SMILES"], as_index=False).agg(
        n_oof_seeds=("split_seed", "nunique"), dvi_11_pred_L_per_kg=("dvi_11_pred_L_per_kg", "mean"),
        partial_vss_11_Rb1_L_per_kg=("partial_vss_11_Rb1_L_per_kg", "mean"),
    )
    long_df = pd.read_csv(STUDY_ROOT / "data" / "processed" / "rat_kp_long.csv", encoding="utf-8-sig")
    observed = long_df.copy(); observed["volume_ml"] = observed["Tissue"].map(volumes)
    complete_ids = observed.groupby("identity_unit_id")["Tissue"].nunique(); complete_ids = complete_ids[complete_ids.eq(11)].index
    complete = observed[observed["identity_unit_id"].isin(complete_ids)].copy()
    complete["component"] = complete["volume_ml"] * complete["Kp"] / bw
    reference = complete.groupby(["parent_group_id", "identity_unit_id", "Drug", "SMILES"], as_index=False)["component"].sum().rename(columns={"component": "dvi_11_true_L_per_kg"})
    complete_eval = dvi_avg.merge(reference, on=["parent_group_id", "identity_unit_id", "Drug", "SMILES"], how="inner")
    complete_eval["y_true_log10_dvi"] = np.log10(complete_eval["dvi_11_true_L_per_kg"])
    complete_eval["y_pred_log10_dvi"] = np.log10(complete_eval["dvi_11_pred_L_per_kg"])
    complete_eval["residual_log10"] = complete_eval["y_pred_log10_dvi"] - complete_eval["y_true_log10_dvi"]
    return grid, dvi_avg, complete_eval, pd.DataFrame(recon_rows)


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    residual = frame["residual_log10"].to_numpy(float)
    absolute = np.abs(residual)
    return {
        "n": len(frame), "n_parent_groups": frame["parent_group_id"].nunique() if "parent_group_id" in frame else len(frame),
        "rmse_log10": float(np.sqrt(np.mean(residual ** 2))), "mae_log10": float(np.mean(absolute)),
        "median_absolute_fold_error": float(np.median(10 ** absolute)),
        "within_2fold": float(np.mean(absolute <= math.log10(2))),
        "within_3fold": float(np.mean(absolute <= math.log10(3))),
        "within_4fold": float(np.mean(absolute <= math.log10(4))),
    }


def summarize_metrics(observed_avg: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_type, model, method), frame in observed_avg[observed_avg["n_tissues"] >= 6].groupby(["split_type", "model", "method"]):
        rows.append({"analysis": "observed_panel_ge6", "split_type": split_type, "model": model, "method": method, **_metrics(frame)})
    for method, frame in complete.groupby("method"):
        rows.append({"analysis": "complete_11_tissue_exploratory", "split_type": "parent_group", "model": "gine", "method": method, **_metrics(frame)})
    return pd.DataFrame(rows)


def paired_context_effects(observed_avg: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(); rng = np.random.default_rng(int(cfg["bootstrap_seed"])); n_boot = int(cfg["bootstrap_replicates"])
    rows = []
    keys = ["parent_group_id", "identity_unit_id"]
    eligible = observed_avg[observed_avg["n_tissues"] >= 6].copy()
    for split_type, split_frame in eligible.groupby("split_type"):
        full = split_frame[split_frame["method"].eq("full_context")][keys + ["residual_log10"]].rename(columns={"residual_log10": "full"})
        for comparator in ("structure_only", "additive_tissue_intercept"):
            comp = split_frame[split_frame["method"].eq(comparator)][keys + ["residual_log10"]].rename(columns={"residual_log10": "comp"})
            paired = full.merge(comp, on=keys, validate="one_to_one")
            groups = paired["parent_group_id"].drop_duplicates().to_numpy()
            def effects(x: pd.DataFrame) -> tuple[float, float]:
                return (
                    float(np.sqrt(np.mean(x["full"] ** 2)) - np.sqrt(np.mean(x["comp"] ** 2))),
                    float(np.mean(np.abs(x["full"])) - np.mean(np.abs(x["comp"]))),
                )
            point = effects(paired)
            cluster = paired.groupby("parent_group_id", as_index=False).agg(
                n=("identity_unit_id", "size"), full_sq=("full", lambda x: float(np.sum(x ** 2))),
                comp_sq=("comp", lambda x: float(np.sum(x ** 2))),
                full_abs=("full", lambda x: float(np.sum(np.abs(x)))),
                comp_abs=("comp", lambda x: float(np.sum(np.abs(x)))),
            )
            draw = rng.integers(0, len(cluster), size=(n_boot, len(cluster)))
            n = cluster["n"].to_numpy(float)[draw].sum(axis=1)
            full_sq = cluster["full_sq"].to_numpy(float)[draw].sum(axis=1)
            comp_sq = cluster["comp_sq"].to_numpy(float)[draw].sum(axis=1)
            full_abs = cluster["full_abs"].to_numpy(float)[draw].sum(axis=1)
            comp_abs = cluster["comp_abs"].to_numpy(float)[draw].sum(axis=1)
            boot = np.column_stack((np.sqrt(full_sq / n) - np.sqrt(comp_sq / n), full_abs / n - comp_abs / n))
            rows.append({
                "split_type": split_type, "comparison": f"full_context_minus_{comparator}",
                "n_identity_units": len(paired), "n_parent_groups": len(groups),
                "delta_rmse_log10": point[0], "ci95_delta_rmse_low": float(np.quantile(boot[:, 0], .025)),
                "ci95_delta_rmse_high": float(np.quantile(boot[:, 0], .975)),
                "delta_mae_log10": point[1], "ci95_delta_mae_low": float(np.quantile(boot[:, 1], .025)),
                "ci95_delta_mae_high": float(np.quantile(boot[:, 1], .975)),
            })
    return pd.DataFrame(rows)


def build_literature_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(); volumes = cfg["tissue_volumes_ml"]; bw = float(cfg["body_weight_g"])
    path = STUDY_ROOT / "injection_study" / "results" / "analysis" / "separate_literature_benchmarks" / "direct_panel_record_predictions.csv"
    data = pd.read_csv(path, encoding="utf-8-sig")
    data["volume_ml"] = data["Tissue_paper"].map(volumes)
    if data["volume_ml"].isna().any():
        raise RuntimeError("Literature panel contains unmapped tissues")
    data["true_component"] = data["volume_ml"] * data["paper_exp_kp"] / bw
    data["baseline_component"] = data["volume_ml"] * data["baseline_pred_kp"] / bw
    data["gnn_component"] = data["volume_ml"] * data["gnn_pred_kp"] / bw
    keys = ["benchmark_method", "split_type", "model", "condition", "workbook_drug", "canonical_smiles"]
    agg = data.groupby(keys, as_index=False).agg(
        n_tissues=("Tissue_paper", "nunique"), tissues=("Tissue_paper", lambda x: "|".join(sorted(set(x)))),
        observed_dvi_true_L_per_kg=("true_component", "sum"), baseline_dvi_pred_L_per_kg=("baseline_component", "sum"),
        gnn_dvi_pred_L_per_kg=("gnn_component", "sum"),
    )
    rows = []
    for predictor, column in [("traditional_equation", "baseline_dvi_pred_L_per_kg"), ("gnn", "gnn_dvi_pred_L_per_kg")]:
        x = agg.copy(); x["predictor"] = predictor; x["observed_dvi_pred_L_per_kg"] = x[column]
        x["residual_log10"] = np.log10(x["observed_dvi_pred_L_per_kg"]) - np.log10(x["observed_dvi_true_L_per_kg"])
        rows.append(x)
    long = pd.concat(rows, ignore_index=True)
    metric_rows = []
    for threshold in (2, 6):
        for keys_value, frame in long[long["n_tissues"] >= threshold].groupby(["benchmark_method", "split_type", "model", "condition", "predictor"]):
            metric_rows.append({"minimum_tissues": threshold, **dict(zip(["benchmark_method", "split_type", "model", "condition", "predictor"], keys_value)), **_metrics(frame)})
    return long, pd.DataFrame(metric_rows)


def run_analysis() -> dict[str, pd.DataFrame]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    seed_identity, observed_avg = build_observed_panel()
    grid, dvi_avg, complete, reconstruction = build_full_grid()
    summary = summarize_metrics(observed_avg, complete)
    paired = paired_context_effects(observed_avg)
    lit, lit_metrics = build_literature_panel()
    grid["contribution_share"] = grid["dvi_component_L_per_kg"] / grid.groupby(["split_seed", "method", "identity_unit_id"])["dvi_component_L_per_kg"].transform("sum")
    tissue_summary = grid.groupby(["method", "Tissue"], as_index=False).agg(
        median_contribution_share=("contribution_share", "median"),
        q25_contribution_share=("contribution_share", lambda x: x.quantile(.25)),
        q75_contribution_share=("contribution_share", lambda x: x.quantile(.75)),
    )
    outputs = {
        "observed_panel_seed_identity.csv": seed_identity,
        "observed_panel_identity_averaged.csv": observed_avg,
        "full_grid_tissue_predictions.csv": grid,
        "full_grid_identity_averaged.csv": dvi_avg,
        "complete_11_tissue_exploratory.csv": complete,
        "vss_propagation_metrics.csv": summary,
        "observed_panel_paired_context_effects.csv": paired,
        "additive_reconstruction_audit.csv": reconstruction,
        "tissue_contribution_summary.csv": tissue_summary,
        "literature_panel_vss_propagation.csv": lit,
        "literature_panel_vss_metrics.csv": lit_metrics,
    }
    for name, frame in outputs.items():
        frame.to_csv(ANALYSIS_DIR / name, index=False, encoding="utf-8-sig")
    return outputs
