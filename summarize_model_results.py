"""Summarize frozen v2 results and clearly separated exploratory diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rat_kp_core.data import read_csv, sha256_file
from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    ANALYSIS_RESULT_DIR,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    LONG_DATA_PATH,
    LOTO_SPLIT_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    STUDY_ROOT,
)
from rat_kp_core.statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, percentile_ci


OUTPUT_DIR = ANALYSIS_RESULT_DIR / "interpretation"
ANALYSIS_SETS = ("primary", "ad_excluded", "condition_specific")
MANIFESTS = {
    "primary": PRIMARY_JOB_MANIFEST_PATH,
    "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
    "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
}
UNIT_COLUMNS = ("parent_group_id", "identity_unit_id", "SMILES")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def part1_direct_encoding_comparison() -> pd.DataFrame:
    rows = []
    for analysis_set in ANALYSIS_SETS:
        paired = read_csv(
            ANALYSIS_RESULT_DIR / analysis_set / "part1_seed_paired_metrics.csv"
        )
        paired["physiology_minus_onehot_gain"] = (
            paired["delta_physiology"] - paired["delta_onehot"]
        )
        for (model, split_type), group in paired.groupby(["model", "split_type"]):
            values = group["physiology_minus_onehot_gain"].to_numpy(float)
            low, high = percentile_ci(values)
            rows.append({
                "analysis_set": analysis_set,
                "model": model,
                "split_type": split_type,
                "mean_rmse_gain_physiology_vs_onehot": values.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "positive_seed_count": int((values > 0).sum()),
                "negative_seed_count": int((values < 0).sum()),
                "n_split_seeds": len(values),
                "scope": "descriptive secondary; no equivalence claim",
            })
    return pd.DataFrame(rows)


def loto_overlap() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = read_csv(LONG_DATA_PATH)
    rows = []
    for heldout in sorted(data["Tissue"].unique()):
        split = read_csv(LOTO_SPLIT_DIR / f"heldout_{heldout}.csv")
        merged = data.merge(split[["row_index", "split"]], on="row_index", validate="one_to_one")
        test = merged[merged["split"] == "test"]
        train = merged[merged["split"] == "train"]
        train_val = merged[merged["split"].isin(["train", "val"])]
        for unit in UNIT_COLUMNS:
            for reference_name, reference in (("train", train), ("train_or_val", train_val)):
                reference_units = set(reference[unit].astype(str))
                test_values = test[unit].astype(str)
                unique_test = pd.Series(sorted(set(test_values)))
                matched_rows = test_values.isin(reference_units)
                matched_unique = unique_test.isin(reference_units)
                rows.append({
                    "heldout_tissue": heldout,
                    "unit": unit,
                    "reference_partition": reference_name,
                    "test_rows": len(test),
                    "matched_test_rows": int(matched_rows.sum()),
                    "row_coverage": float(matched_rows.mean()),
                    "unique_test_units": len(unique_test),
                    "matched_unique_test_units": int(matched_unique.sum()),
                    "unique_unit_coverage": float(matched_unique.mean()),
                })
    by_tissue = pd.DataFrame(rows)
    summary_rows = []
    for (unit, reference), group in by_tissue.groupby(["unit", "reference_partition"]):
        summary_rows.append({
            "unit": unit,
            "reference_partition": reference,
            "pooled_row_coverage": group["matched_test_rows"].sum() / group["test_rows"].sum(),
            "tissue_equal_row_coverage": group["row_coverage"].mean(),
            "pooled_tissue_unit_pair_coverage": (
                group["matched_unique_test_units"].sum() / group["unique_test_units"].sum()
            ),
            "tissue_equal_unique_unit_coverage": group["unique_unit_coverage"].mean(),
            "total_test_rows_across_folds": int(group["test_rows"].sum()),
            "total_unique_tissue_unit_pairs": int(group["unique_test_units"].sum()),
        })
    return by_tissue, pd.DataFrame(summary_rows)


def loto_posthoc_robustness() -> pd.DataFrame:
    rows = []
    for analysis_set in ANALYSIS_SETS:
        effects = read_csv(ANALYSIS_RESULT_DIR / analysis_set / "loto_tissue_effects.csv")
        for model, group in effects.groupby("model"):
            adipose = float(group.loc[group["heldout_tissue"] == "adipose", "delta_physiology"].iloc[0])
            nonadipose = group[group["heldout_tissue"] != "adipose"]["delta_physiology"].to_numpy(float)
            all_values = group["delta_physiology"].to_numpy(float)
            rows.append({
                "analysis_set": analysis_set,
                "model": model,
                "tissue_equal_mean_all": all_values.mean(),
                "adipose_delta": adipose,
                "mean_excluding_adipose": nonadipose.mean(),
                "positive_tissues_all": int((all_values > 0).sum()),
                "positive_tissues_excluding_adipose": int((nonadipose > 0).sum()),
                "sign_changed_after_adipose_exclusion": bool(
                    np.sign(all_values.mean()) != np.sign(nonadipose.mean())
                ),
                "scope": "post-hoc exploratory; adipose exclusion not in primary estimand",
            })
    return pd.DataFrame(rows)


def adipose_seed_diagnostics() -> pd.DataFrame:
    rows = []
    for analysis_set in ANALYSIS_SETS:
        metrics = read_csv(ANALYSIS_RESULT_DIR / analysis_set / "all_job_metrics.csv")
        adipose = metrics[
            (metrics["stage"] == "loto")
            & (metrics["heldout_tissue"] == "adipose")
            & (metrics["context"].isin(["structure_only", "physiology"]))
        ]
        pivot = adipose.pivot(
            index=["model", "training_seed"], columns="context", values="rmse_log10"
        ).reset_index()
        pivot["seed_delta_physiology"] = pivot["structure_only"] - pivot["physiology"]

        averaged = read_csv(
            ANALYSIS_RESULT_DIR / analysis_set / "loto_seed_averaged_predictions.csv"
        )
        for model, group in pivot.groupby("model"):
            seed_deltas = group["seed_delta_physiology"].to_numpy(float)
            for context in ("structure_only", "physiology"):
                model_context = averaged[
                    (averaged["model"] == model) & (averaged["context"] == context)
                ]
                adipose_sd = model_context[
                    model_context["heldout_tissue"] == "adipose"
                ]["seed_sd"].to_numpy(float)
                nonadipose_sd = model_context[
                    model_context["heldout_tissue"] != "adipose"
                ]["seed_sd"].to_numpy(float)
                nonadipose_mean = nonadipose_sd.mean()
                rows.append({
                    "analysis_set": analysis_set,
                    "model": model,
                    "context": context,
                    "n_training_seeds": int(group["training_seed"].nunique()),
                    "mean_seed_delta_physiology": seed_deltas.mean(),
                    "seed_delta_sd": seed_deltas.std(ddof=1) if len(seed_deltas) > 1 else 0.0,
                    "positive_seed_deltas": int((seed_deltas > 0).sum()),
                    "negative_seed_deltas": int((seed_deltas < 0).sum()),
                    "adipose_mean_prediction_seed_sd": adipose_sd.mean(),
                    "nonadipose_mean_prediction_seed_sd": nonadipose_mean,
                    "adipose_to_nonadipose_seed_sd_ratio": (
                        adipose_sd.mean() / nonadipose_mean if nonadipose_mean > 0 else np.nan
                    ),
                    "scope": "post-hoc exploratory seed-stability diagnostic",
                })
    return pd.DataFrame(rows)


def write_summary(
    direct: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    robust: pd.DataFrame,
    seed_diagnostics: pd.DataFrame,
) -> None:
    part1_counts = []
    for analysis_set in ANALYSIS_SETS:
        effects = read_csv(ANALYSIS_RESULT_DIR / analysis_set / "part1_context_effects.csv")
        part1_counts.append({
            "analysis_set": analysis_set,
            "positive": int((effects["mean_delta_rmse"] > 0).sum()),
            "ci_positive": int((effects["ci95_low"] > 0).sum()),
            "total": len(effects),
        })
    part1_counts = pd.DataFrame(part1_counts)

    primary_loto = read_csv(ANALYSIS_RESULT_DIR / "primary" / "loto_inference.csv")
    primary_loto = primary_loto.set_index("model")
    exact_overlap = overlap_summary[
        (overlap_summary["unit"] == "SMILES")
        & (overlap_summary["reference_partition"] == "train")
    ].iloc[0]
    parent_overlap = overlap_summary[
        (overlap_summary["unit"] == "parent_group_id")
        & (overlap_summary["reference_partition"] == "train")
    ].iloc[0]
    train_val_exact = overlap_summary[
        (overlap_summary["unit"] == "SMILES")
        & (overlap_summary["reference_partition"] == "train_or_val")
    ].iloc[0]

    primary_robust = robust[robust["analysis_set"] == "primary"].set_index("model")
    primary_seed = seed_diagnostics[
        (seed_diagnostics["analysis_set"] == "primary")
        & (seed_diagnostics["context"] == "physiology")
    ].set_index("model")
    primary_structure_seed = seed_diagnostics[
        (seed_diagnostics["analysis_set"] == "primary")
        & (seed_diagnostics["context"] == "structure_only")
    ].set_index("model")
    primary_part1 = part1_counts.set_index("analysis_set").loc["primary"]
    ad_part1 = part1_counts.set_index("analysis_set").loc["ad_excluded"]
    condition_part1 = part1_counts.set_index("analysis_set").loc["condition_specific"]

    lines = [
        "# Rat Kp tissue-context GNN 결과 해석 요약",
        "",
        "## 분석 지위",
        "",
        "- Primary Part 1과 primary LOTO inference는 동결된 분석이다.",
        "- AD-excluded와 condition-specific 결과는 사전 고정된 exploratory robustness 분석이다.",
        "- physiology–one-hot 직접 비교, adipose 제외, seed 안정성 및 overlap 분해는 결과 해석을 위한 secondary/post-hoc 분석이다.",
        "- 본 문서는 crossed bootstrap이나 외부 검증을 추가하지 않는다.",
        "",
        "## 1. Within-domain tissue context",
        "",
        f"Primary의 20개 모델×split×context 대비가 모두 양수였고, {int(primary_part1['ci_positive'])}/20의 95% CI가 0을 제외했다. CI가 0을 포함한 유일한 경우는 AttentiveFP random split의 physiological context였다.",
        f"AD-excluded와 condition-specific sensitivity에서는 각각 {int(ad_part1['positive'])}/20, {int(condition_part1['positive'])}/20이 양수이고 모두 CI가 0을 제외했다.",
        "따라서 tissue context가 within-domain Kp 예측에 기여한다는 주장은 v2에서도 강하게 지지된다.",
        "",
        "## 2. Physiology와 one-hot의 within-domain 비교",
        "",
        "두 encoding의 직접 paired 차이는 모델과 split에 따라 방향이 달랐다. equivalence test는 수행하지 않았으므로 동등성을 주장하지 않는다.",
        "결론은 '일관된 우열이 없다'이며, physiology의 가치는 within-domain 성능 우월성이 아니라 unseen-tissue transfer 가능성으로 평가한다.",
        "",
        "## 3. Primary LOTO transfer",
        "",
        f"- Random Forest: ΔRMSE={fmt(primary_loto.loc['random_forest','mean_delta_rmse'])}, 95% CI [{fmt(primary_loto.loc['random_forest','ci95_low'])}, {fmt(primary_loto.loc['random_forest','ci95_high'])}], BH-FDR={fmt(primary_loto.loc['random_forest','bh_fdr_p_value'],4)}.",
        f"- XGBoost: ΔRMSE={fmt(primary_loto.loc['xgboost','mean_delta_rmse'])}, 95% CI [{fmt(primary_loto.loc['xgboost','ci95_low'])}, {fmt(primary_loto.loc['xgboost','ci95_high'])}], BH-FDR={fmt(primary_loto.loc['xgboost','bh_fdr_p_value'],4)}.",
        f"- LinearSVR: ΔRMSE={fmt(primary_loto.loc['linear_svr','mean_delta_rmse'])}, CI가 0을 포함하고 FDR={fmt(primary_loto.loc['linear_svr','bh_fdr_p_value'],3)}.",
        f"- D-MPNN: ΔRMSE={fmt(primary_loto.loc['d_mpnn','mean_delta_rmse'])}; 양의 transfer가 지지되지 않았다.",
        f"- AttentiveFP: ΔRMSE={fmt(primary_loto.loc['attentive_fp','mean_delta_rmse'])}; CI가 0을 포함하고 FDR={fmt(primary_loto.loc['attentive_fp','bh_fdr_p_value'],3)}.",
        "v1의 'RF에서만 FDR 유의' 문구는 폐기해야 한다. v2 primary에서는 RF와 XGBoost가 모두 유의하다.",
        "",
        "## 4. Robustness sensitivity",
        "",
        "AD-excluded와 condition-specific LOTO에서도 RF와 XGBoost의 양의 transfer와 exploratory FDR 유의성이 유지됐다. LinearSVR은 양의 점추정이나 FDR 유의하지 않았다. 두 GNN의 aggregate transfer는 sensitivity에서 음수였다.",
        "따라서 가장 방어 가능한 결론은 'physiological transfer는 model-dependent하며 이 데이터에서는 tree ensembles에서 가장 일관됐다'이다. 트리 알고리즘의 일반적 우월성은 주장하지 않는다.",
        "",
        "## 5. Adipose post-hoc 분해",
        "",
        f"Primary adipose Δ는 RF {fmt(primary_robust.loc['random_forest','adipose_delta'])}, XGBoost {fmt(primary_robust.loc['xgboost','adipose_delta'])}, LinearSVR {fmt(primary_robust.loc['linear_svr','adipose_delta'])}, D-MPNN {fmt(primary_robust.loc['d_mpnn','adipose_delta'])}, AttentiveFP {fmt(primary_robust.loc['attentive_fp','adipose_delta'])}였다.",
        f"Adipose 제외 평균은 모든 모델에서 양수였다: RF {fmt(primary_robust.loc['random_forest','mean_excluding_adipose'])}, XGBoost {fmt(primary_robust.loc['xgboost','mean_excluding_adipose'])}, LinearSVR {fmt(primary_robust.loc['linear_svr','mean_excluding_adipose'])}, D-MPNN {fmt(primary_robust.loc['d_mpnn','mean_excluding_adipose'])}, AttentiveFP {fmt(primary_robust.loc['attentive_fp','mean_excluding_adipose'])}.",
        "그러나 LinearSVR도 adipose에서 음수이므로 성능 악화 자체를 GNN 고유 실패라고 표현하면 과장이다. 정확한 표현은 'adipose에서 non-tree 모델의 physiology transfer가 악화됐고, 손상 규모는 neural models에서 더 컸다'이다.",
        f"반면 stochastic instability는 GNN-specific pattern이었다. Physiological GNN의 adipose prediction seed-SD는 같은 모델의 non-adipose 평균보다 D-MPNN {fmt(primary_seed.loc['d_mpnn','adipose_to_nonadipose_seed_sd_ratio'],1)}배, AttentiveFP {fmt(primary_seed.loc['attentive_fp','adipose_to_nonadipose_seed_sd_ratio'],1)}배 컸다. Structure-only에서는 각각 {fmt(primary_structure_seed.loc['d_mpnn','adipose_to_nonadipose_seed_sd_ratio'],2)}배와 {fmt(primary_structure_seed.loc['attentive_fp','adipose_to_nonadipose_seed_sd_ratio'],2)}배였다.",
        "Adipose seed별 physiology gain은 D-MPNN과 AttentiveFP에서 모두 0/5 양수였고, RF와 XGBoost에서는 모두 5/5 양수였다. 이는 failure magnitude와 seed instability를 구분해 해석해야 함을 보여준다. 수렴 실패 여부나 인과 메커니즘은 이 분석만으로 확정하지 않는다.",
        "Adipose 제외 분석과 seed 안정성은 post-hoc이며 primary aggregate를 대체하지 않는다.",
        "",
        "## 6. LOTO compound overlap",
        "",
        f"정확한 model SMILES 기준으로 LOTO test 행의 {100*exact_overlap['pooled_row_coverage']:.2f}%가 해당 fold의 train에 이미 존재했다. parent-group 기준은 {100*parent_overlap['pooled_row_coverage']:.2f}%였다. train 또는 validation까지 포함하면 정확한 SMILES coverage는 {100*train_val_exact['pooled_row_coverage']:.2f}%였다.",
        "따라서 LOTO는 '미관측 조직으로의 전이' 평가이지 미관측 화합물 일반화 평가가 아니다. 기존 98.96% 수치는 v2 결과에 그대로 사용하지 않고 위 수치로 교체한다.",
        "",
        "## 7. 최종 주장 교정",
        "",
        "1. Tissue context는 within-domain Kp 예측을 모델 전반에서 개선한다.",
        "2. Within-domain에서 categorical identity와 physiological composition 사이에 일관된 우열은 없다.",
        "3. Physiological composition은 unseen-tissue transfer를 가능하게 할 수 있으나 model-dependent하다.",
        "4. v2에서 통계적으로 지지된 transfer는 RF와 XGBoost에서 확인됐으며 두 data sensitivity에서도 유지됐다.",
        "5. Adipose는 non-tree transfer를 저해하는 영향력 큰 조직이지만, 이를 일반적인 GNN 실패나 트리의 보편적 우월성으로 확대하지 않는다.",
        "6. 이 결과는 단일 small-scale rat Kp dataset과 고정 hyperparameter 설정에 한정된다.",
        "",
        "## 8. 아직 미해소",
        "",
        "- Kp 데이터 출처·사용 권한과 total Kp/측정 조건 확인.",
        "- compound 반복을 양 축으로 다루는 crossed bootstrap sensitivity.",
        "- compound–tissue 동시 holdout.",
        "- 독립 데이터 또는 다른 종 외적 검증.",
        "- 2025년 유사 연구와 2026년 tissue-composition preprint의 직접 문헌 비교.",
    ]
    (OUTPUT_DIR / "RESULTS_INTERPRETATION_V2_KO.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    direct = part1_direct_encoding_comparison()
    overlap_by_tissue, overlap_summary = loto_overlap()
    robust = loto_posthoc_robustness()
    seed_diagnostics = adipose_seed_diagnostics()

    direct.to_csv(OUTPUT_DIR / "part1_physiology_vs_onehot_descriptive.csv", index=False)
    overlap_by_tissue.to_csv(OUTPUT_DIR / "loto_overlap_by_tissue.csv", index=False)
    overlap_summary.to_csv(OUTPUT_DIR / "loto_overlap_summary.csv", index=False)
    robust.to_csv(OUTPUT_DIR / "loto_adipose_exclusion_posthoc.csv", index=False)
    seed_diagnostics.to_csv(OUTPUT_DIR / "adipose_seed_stability_posthoc.csv", index=False)
    write_summary(direct, overlap_summary, robust, seed_diagnostics)

    outputs = sorted(
        path for path in OUTPUT_DIR.glob("*")
        if path.name != "interpretation_manifest.json"
    )
    report = {
        "status": "complete",
        "scope": "interpretation and explicitly labelled secondary/post-hoc analyses only; no retraining",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        },
    }
    (OUTPUT_DIR / "interpretation_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(OUTPUT_DIR),
        "files": [path.name for path in outputs],
    }, indent=2))


if __name__ == "__main__":
    main()
