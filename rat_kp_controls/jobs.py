"""Build the fixed 520-job plan for the two targeted additional analyses."""

from __future__ import annotations

import pandas as pd

from rat_kp_core.data import read_csv
from rat_kp_core.paths import LONG_DATA_PATH, SPLIT_DIR, STUDY_ROOT

from .config import load_config
from .paths import MANIFEST_DIR, PLAN_PATH


MODELS = ("gcn", "gine", "d_mpnn", "attentive_fp")


def _relative(path) -> str:
    return str(path.relative_to(STUDY_ROOT))


def _base_row(
    *,
    analysis: str,
    stage: str,
    model: str,
    condition: str,
    context_type: str,
    injection_position: str,
    physiology_scaling: str,
    split_type: str,
    split_seed: int,
    heldout_tissue: str,
    training_seed: int,
    split_path,
) -> dict:
    job_id = (
        f"addctx__{analysis}__{split_type}__split{split_seed}__"
        f"{heldout_tissue or 'none'}__{model}__{condition}__train{training_seed}"
    )
    return {
        "job_id": job_id,
        "analysis": analysis,
        "stage": stage,
        "analysis_set": "primary",
        "dataset_key": "primary",
        "dataset_path": _relative(LONG_DATA_PATH),
        "model": model,
        "condition": condition,
        "context_type": context_type,
        "injection_position": injection_position,
        "physiology_scaling": physiology_scaling,
        "split_type": split_type,
        "split_seed": split_seed,
        "heldout_tissue": heldout_tissue,
        "training_seed": training_seed,
        "split_path": _relative(split_path),
        "execution_mode": "new",
    }


def _additive_rows() -> list[dict]:
    rows = []
    for split_type in ("parent_group", "scaffold"):
        internal = "random" if split_type == "parent_group" else "scaffold"
        for split_seed in range(10):
            split_path = SPLIT_DIR / f"{internal}_seed{split_seed}.csv"
            for model in MODELS:
                rows.append(_base_row(
                    analysis="primary_context_decomposition",
                    stage="part1",
                    model=model,
                    condition="additive_tissue_intercept",
                    context_type="tissue_onehot",
                    injection_position="additive",
                    physiology_scaling="not_applicable",
                    split_type=split_type,
                    split_seed=split_seed,
                    heldout_tissue="",
                    training_seed=split_seed,
                    split_path=split_path,
                ))
    return rows


def _raw_loto_rows() -> list[dict]:
    tissues = sorted(read_csv(LONG_DATA_PATH)["Tissue"].astype(str).unique())
    rows = []
    for tissue in tissues:
        split_path = SPLIT_DIR / "loto" / f"heldout_{tissue}.csv"
        for model in MODELS:
            for training_seed in range(5):
                for position in ("late", "early"):
                    rows.append(_base_row(
                        analysis="loto_raw_fraction_robustness",
                        stage="loto",
                        model=model,
                        condition=f"physiology_{position}",
                        context_type="physiology",
                        injection_position=position,
                        physiology_scaling="raw_fraction",
                        split_type="loto",
                        split_seed=0,
                        heldout_tissue=tissue,
                        training_seed=training_seed,
                        split_path=split_path,
                    ))
    return rows


def build_plan() -> pd.DataFrame:
    config = load_config()
    if tuple(config["models"]) != MODELS:
        raise RuntimeError("Additional-analysis model order mismatch")
    plan = pd.DataFrame(_additive_rows() + _raw_loto_rows())
    if len(plan) != 520 or plan["job_id"].duplicated().any():
        raise RuntimeError("Invalid additional-analysis job plan")
    observed = plan.groupby("analysis").size().to_dict()
    expected = {
        "primary_context_decomposition": 80,
        "loto_raw_fraction_robustness": 440,
    }
    if observed != expected:
        raise RuntimeError(f"Unexpected additional-analysis matrix: {observed}")
    return plan


def write_plan() -> pd.DataFrame:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    plan.to_csv(PLAN_PATH, index=False, encoding="utf-8-sig")
    return plan

