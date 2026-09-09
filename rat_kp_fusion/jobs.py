"""Build the immutable 1,860-job plan without launching model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rat_kp_core.data import read_csv
from rat_kp_core.paths import (
    AD_SENSITIVITY_DATA_PATH,
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_DATA_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    LONG_DATA_PATH,
    PRIMARY_JOB_MANIFEST_PATH,
    SPLIT_DIR,
    STUDY_ROOT,
)

from .config import load_config
from .paths import MANIFEST_DIR, NEW_JOB_MANIFEST_PATH, PLAN_MANIFEST_PATH, REUSE_MANIFEST_PATH


MODELS = ("gcn", "gine", "d_mpnn", "attentive_fp")
PART1_CONDITIONS = (
    "structure_only",
    "onehot_late",
    "onehot_early",
    "physiology_late",
    "physiology_early",
)
LOTO_CONDITIONS = ("structure_only", "physiology_late", "physiology_early")
CONDITION_MAP = {
    "structure_only": ("structure_only", "none"),
    "onehot_late": ("tissue_onehot", "late"),
    "onehot_early": ("tissue_onehot", "early"),
    "physiology_late": ("physiology", "late"),
    "physiology_early": ("physiology", "early"),
}
LEGACY_CONTEXT_MAP = {
    "structure_only": "structure_only",
    "onehot_late": "tissue_onehot",
    "physiology_late": "physiology",
}
DATASETS = {
    "primary": (LONG_DATA_PATH, SPLIT_DIR),
    "ad_excluded": (
        AD_SENSITIVITY_DATA_PATH,
        STUDY_ROOT / "data" / "splits_sensitivity" / "ad_excluded",
    ),
    "condition_specific": (
        CONDITION_SENSITIVITY_DATA_PATH,
        STUDY_ROOT / "data" / "splits_sensitivity" / "condition_specific",
    ),
}
LEGACY_MANIFESTS = {
    "primary": PRIMARY_JOB_MANIFEST_PATH,
    "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
    "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
}


def _relative(path: Path) -> str:
    return str(path.relative_to(STUDY_ROOT))


def _execution_mode(model: str, condition: str) -> str:
    if model in {"d_mpnn", "attentive_fp"} and condition in LEGACY_CONTEXT_MAP:
        return "legacy_reuse"
    return "new"


def _base_row(
    *,
    analysis_set: str,
    stage: str,
    model: str,
    condition: str,
    split_type: str,
    split_seed: int,
    heldout_tissue: str,
    training_seed: int,
    dataset_path: Path,
    split_path: Path,
) -> dict:
    context_type, injection_position = CONDITION_MAP[condition]
    job_id = (
        f"inject__{stage}__{analysis_set}__{split_type}__split{split_seed}__"
        f"{heldout_tissue or 'none'}__{model}__{condition}__train{training_seed}"
    )
    return {
        "job_id": job_id,
        "stage": stage,
        "analysis_set": analysis_set,
        "dataset_key": analysis_set,
        "dataset_path": _relative(dataset_path),
        "model": model,
        "condition": condition,
        "context_type": context_type,
        "injection_position": injection_position,
        "physiology_scaling": "tissue_equal",
        "split_type": split_type,
        "split_seed": split_seed,
        "heldout_tissue": heldout_tissue,
        "training_seed": training_seed,
        "split_path": _relative(split_path),
        "execution_mode": _execution_mode(model, condition),
        "legacy_job_id": "",
        "legacy_result_path": "",
    }


def _part1_rows(analysis_set: str) -> list[dict]:
    dataset_path, split_dir = DATASETS[analysis_set]
    rows = []
    for split_type in ("parent_group", "scaffold"):
        internal_split = "random" if split_type == "parent_group" else "scaffold"
        for split_seed in range(10):
            split_path = split_dir / f"{internal_split}_seed{split_seed}.csv"
            for model in MODELS:
                for condition in PART1_CONDITIONS:
                    rows.append(_base_row(
                        analysis_set=analysis_set,
                        stage="part1",
                        model=model,
                        condition=condition,
                        split_type=split_type,
                        split_seed=split_seed,
                        heldout_tissue="",
                        training_seed=split_seed,
                        dataset_path=dataset_path,
                        split_path=split_path,
                    ))
    return rows


def _loto_rows() -> list[dict]:
    dataset_path, split_dir = DATASETS["primary"]
    tissues = sorted(read_csv(dataset_path)["Tissue"].astype(str).unique())
    rows = []
    for tissue in tissues:
        split_path = split_dir / "loto" / f"heldout_{tissue}.csv"
        for model in MODELS:
            for training_seed in range(5):
                for condition in LOTO_CONDITIONS:
                    rows.append(_base_row(
                        analysis_set="primary",
                        stage="loto",
                        model=model,
                        condition=condition,
                        split_type="loto",
                        split_seed=0,
                        heldout_tissue=tissue,
                        training_seed=training_seed,
                        dataset_path=dataset_path,
                        split_path=split_path,
                    ))
    return rows


def _attach_legacy_mapping(plan: pd.DataFrame) -> pd.DataFrame:
    legacy_frames = []
    for analysis_set, path in LEGACY_MANIFESTS.items():
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        frame = frame[frame["model"].isin(["d_mpnn", "attentive_fp"])].copy()
        frame["analysis_set_lookup"] = analysis_set
        legacy_frames.append(frame)
    legacy = pd.concat(legacy_frames, ignore_index=True)

    lookup = {}
    for row in legacy.itertuples(index=False):
        external_split = "parent_group" if row.split_type == "random" else row.split_type
        key = (
            row.analysis_set_lookup,
            row.stage,
            row.model,
            row.context,
            external_split,
            int(row.split_seed),
            str(row.heldout_tissue),
            int(row.training_seed),
        )
        result_path = (
            STUDY_ROOT / "results" / "experiments" / row.analysis_set / row.stage / row.job_id
        )
        lookup[key] = (row.job_id, _relative(result_path))

    result = plan.copy()
    for index, row in result[result["execution_mode"] == "legacy_reuse"].iterrows():
        key = (
            row.analysis_set,
            row.stage,
            row.model,
            LEGACY_CONTEXT_MAP[row.condition],
            row.split_type,
            int(row.split_seed),
            str(row.heldout_tissue),
            int(row.training_seed),
        )
        if key not in lookup:
            raise RuntimeError(f"Missing legacy mapping for {row.job_id}")
        legacy_job_id, legacy_result_path = lookup[key]
        result.at[index, "legacy_job_id"] = legacy_job_id
        result.at[index, "legacy_result_path"] = legacy_result_path
    return result


def build_plan() -> pd.DataFrame:
    config = load_config()
    if tuple(config["models"]) != MODELS:
        raise RuntimeError("Config/model order mismatch")
    rows = _part1_rows("primary") + _loto_rows()
    rows += _part1_rows("ad_excluded") + _part1_rows("condition_specific")
    plan = _attach_legacy_mapping(pd.DataFrame(rows))
    if len(plan) != 1860 or plan["job_id"].duplicated().any():
        raise RuntimeError("Invalid 1,860-job plan")
    expected = {
        ("primary", "part1"): 400,
        ("primary", "loto"): 660,
        ("ad_excluded", "part1"): 400,
        ("condition_specific", "part1"): 400,
    }
    observed = plan.groupby(["analysis_set", "stage"]).size().to_dict()
    if observed != expected:
        raise RuntimeError(f"Unexpected plan matrix: {observed}")
    counts = plan["execution_mode"].value_counts().to_dict()
    if counts != {"new": 1280, "legacy_reuse": 580}:
        raise RuntimeError(f"Unexpected reuse/new counts: {counts}")
    if ((plan["stage"] == "loto") & plan["condition"].str.startswith("onehot")).any():
        raise RuntimeError("One-hot LOTO is prohibited")
    if ((plan["analysis_set"] != "primary") & (plan["stage"] == "loto")).any():
        raise RuntimeError("Sensitivity LOTO is prohibited")
    return plan


def write_plan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    new = plan[plan["execution_mode"] == "new"].reset_index(drop=True)
    reuse = plan[plan["execution_mode"] == "legacy_reuse"].reset_index(drop=True)
    plan.to_csv(PLAN_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    new.to_csv(NEW_JOB_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    reuse.to_csv(REUSE_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return plan, new, reuse
