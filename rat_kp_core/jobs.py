"""Frozen v2 primary and data-robustness job manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import read_csv, sha256_file
from .paths import (
    AD_SENSITIVITY_DATA_PATH,
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_DATA_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    CONFIG_PATH,
    LONG_DATA_PATH,
    LOTO_SPLIT_DIR,
    MANIFEST_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    SPLIT_DIR,
    STUDY_ROOT,
)
from .physiology import CONTEXT_MODES


MODELS = ("random_forest", "xgboost", "linear_svr", "d_mpnn", "attentive_fp")
LOTO_SEEDS = {
    "random_forest": range(5),
    "xgboost": range(5),
    "linear_svr": (0,),
    "d_mpnn": range(5),
    "attentive_fp": range(5),
}
SENSITIVITY_CONTEXTS_PART1 = CONTEXT_MODES
SENSITIVITY_CONTEXTS_LOTO = ("structure_only", "physiology")

DATASETS = {
    "primary": {
        "path": LONG_DATA_PATH,
        "split_dir": SPLIT_DIR,
        "analysis_set": "primary",
    },
    "ad_excluded": {
        "path": AD_SENSITIVITY_DATA_PATH,
        "split_dir": STUDY_ROOT / "data" / "splits_sensitivity" / "ad_excluded",
        "analysis_set": "ad_excluded",
    },
    "condition_specific": {
        "path": CONDITION_SENSITIVITY_DATA_PATH,
        "split_dir": STUDY_ROOT / "data" / "splits_sensitivity" / "condition_specific",
        "analysis_set": "condition_specific",
    },
}


def _part1_jobs(dataset_key: str, contexts: tuple[str, ...]) -> list[dict]:
    spec = DATASETS[dataset_key]
    rows: list[dict] = []
    for split_type in ("random", "scaffold"):
        for split_seed in range(10):
            split_path = spec["split_dir"] / f"{split_type}_seed{split_seed}.csv"
            for model in MODELS:
                training_seed = 0 if model == "linear_svr" else split_seed
                for context in contexts:
                    job_id = (
                        f"part1__{dataset_key}__{split_type}__split{split_seed}__"
                        f"{model}__{context}__train{training_seed}"
                    )
                    rows.append({
                        "job_id": job_id,
                        "stage": "part1",
                        "analysis_set": spec["analysis_set"],
                        "dataset_key": dataset_key,
                        "dataset_path": str(spec["path"].relative_to(STUDY_ROOT)),
                        "model": model,
                        "context": context,
                        "physiology_scaling": "tissue_equal",
                        "split_type": split_type,
                        "split_seed": split_seed,
                        "heldout_tissue": "",
                        "training_seed": training_seed,
                        "split_path": str(split_path.relative_to(STUDY_ROOT)),
                    })
    return rows


def _loto_jobs(dataset_key: str, contexts: tuple[str, ...]) -> list[dict]:
    spec = DATASETS[dataset_key]
    rows: list[dict] = []
    tissues = sorted(read_csv(spec["path"])["Tissue"].unique())
    for heldout in tissues:
        split_path = spec["split_dir"] / "loto" / f"heldout_{heldout}.csv"
        for model in MODELS:
            for training_seed in LOTO_SEEDS[model]:
                for context in contexts:
                    job_id = f"loto__{dataset_key}__{heldout}__{model}__{context}__train{training_seed}"
                    rows.append({
                        "job_id": job_id,
                        "stage": "loto",
                        "analysis_set": spec["analysis_set"],
                        "dataset_key": dataset_key,
                        "dataset_path": str(spec["path"].relative_to(STUDY_ROOT)),
                        "model": model,
                        "context": context,
                        "physiology_scaling": "tissue_equal",
                        "split_type": "loto",
                        "split_seed": 0,
                        "heldout_tissue": heldout,
                        "training_seed": training_seed,
                        "split_path": str(split_path.relative_to(STUDY_ROOT)),
                    })
    return rows


def build_job_manifests() -> dict[str, pd.DataFrame]:
    manifests = {
        "primary": pd.DataFrame(
            _part1_jobs("primary", CONTEXT_MODES)
            + _loto_jobs("primary", CONTEXT_MODES)
        ),
        "ad_excluded": pd.DataFrame(
            _part1_jobs("ad_excluded", SENSITIVITY_CONTEXTS_PART1)
            + _loto_jobs("ad_excluded", SENSITIVITY_CONTEXTS_LOTO)
        ),
        "condition_specific": pd.DataFrame(
            _part1_jobs("condition_specific", SENSITIVITY_CONTEXTS_PART1)
            + _loto_jobs("condition_specific", SENSITIVITY_CONTEXTS_LOTO)
        ),
    }
    expected = {"primary": 993, "ad_excluded": 762, "condition_specific": 762}
    for name, frame in manifests.items():
        if len(frame) != expected[name] or frame["job_id"].duplicated().any():
            raise RuntimeError(f"Invalid {name} job manifest")
        for relative in frame["dataset_path"].unique():
            if not (STUDY_ROOT / relative).exists():
                raise FileNotFoundError(relative)
        for relative in frame["split_path"].unique():
            if not (STUDY_ROOT / relative).exists():
                raise FileNotFoundError(relative)
    return manifests


def write_job_manifests() -> dict:
    manifests = build_job_manifests()
    paths = {
        "primary": PRIMARY_JOB_MANIFEST_PATH,
        "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
        "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in manifests.items():
        frame.to_csv(paths[name], index=False, encoding="utf-8-sig")
    report = {
        "scope": "frozen v2 jobs only; no model execution",
        "fixed_config_sha256": sha256_file(CONFIG_PATH),
        "manifests": {
            name: {
                "jobs": len(frame),
                "part1_jobs": int((frame["stage"] == "part1").sum()),
                "loto_jobs": int((frame["stage"] == "loto").sum()),
                "dataset_sha256": sha256_file(STUDY_ROOT / frame.iloc[0]["dataset_path"]),
                "manifest_sha256": sha256_file(paths[name]),
            }
            for name, frame in manifests.items()
        },
    }
    (MANIFEST_DIR / "job_manifest_audit_v2.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

