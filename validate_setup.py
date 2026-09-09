"""Fail-closed v2 pre-experiment audit; never executes full model jobs."""

from __future__ import annotations

import json
import platform

import chemprop
import lightning
import pandas as pd
import rdkit
import sklearn
import torch
import torch_geometric
import xgboost

from rat_kp_core.data import read_csv, sha256_file
from rat_kp_core.jobs import build_job_manifests
from rat_kp_core.paths import (
    AD_SENSITIVITY_DATA_PATH,
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_DATA_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    CONFIG_PATH,
    EXPERIMENT_RESULT_DIR,
    LONG_DATA_PATH,
    LOTO_SPLIT_DIR,
    MANIFEST_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    SPLIT_DIR,
    STUDY_ROOT,
    TABULAR_FEATURE_CACHE_PATH,
    TABULAR_FEATURE_INDEX_PATH,
)
from rat_kp_core.physiology import ContextEncoder, load_physiology


EXPECTED_DATASETS = {
    "primary": (LONG_DATA_PATH, STUDY_ROOT / "data" / "splits", 1270),
    "ad_excluded": (
        AD_SENSITIVITY_DATA_PATH,
        STUDY_ROOT / "data" / "splits_sensitivity" / "ad_excluded",
        1252,
    ),
    "condition_specific": (
        CONDITION_SENSITIVITY_DATA_PATH,
        STUDY_ROOT / "data" / "splits_sensitivity" / "condition_specific",
        1480,
    ),
}


def _validate_split_set(name, data_path, split_root, expected_rows):
    data = read_csv(data_path)
    if len(data) != expected_rows or data["Tissue"].nunique() != 11:
        raise RuntimeError(f"{name} numerical data contract failed")
    if data["row_index"].nunique() != len(data):
        raise RuntimeError(f"{name} row_index is not unique")

    summaries = []
    hashes = {}
    for split_type in ("random", "scaffold"):
        for seed in range(10):
            path = split_root / f"{split_type}_seed{seed}.csv"
            split = read_csv(path)
            if len(split) != len(data) or split["row_index"].nunique() != len(data):
                raise RuntimeError(f"Invalid row coverage: {path}")
            if set(split["split"]) != {"train", "val", "test"}:
                raise RuntimeError(f"Invalid partition labels: {path}")
            if split.groupby("parent_group_id")["split"].nunique().max() != 1:
                raise RuntimeError(f"Parent-group leakage: {path}")
            hashes[str(path.relative_to(STUDY_ROOT))] = sha256_file(path)

    matrix = load_physiology()
    for heldout in sorted(data["Tissue"].unique()):
        path = split_root / "loto" / f"heldout_{heldout}.csv"
        split = read_csv(path)
        merged = data.merge(
            split[["row_index", "split"]], on="row_index", validate="one_to_one"
        )
        train, val, test = [merged[merged["split"] == label] for label in ("train", "val", "test")]
        if any(frame.empty for frame in (train, val, test)):
            raise RuntimeError(f"Empty LOTO partition: {path}")
        if set(test["Tissue"]) != {heldout} or heldout in set(train["Tissue"]) | set(val["Tissue"]):
            raise RuntimeError(f"Invalid held-out tissue boundary: {path}")
        if set(train["parent_group_id"]) & set(val["parent_group_id"]):
            raise RuntimeError(f"LOTO train/validation parent leakage: {path}")
        onehot = ContextEncoder("tissue_onehot", matrix).fit(train).transform(test)
        physiology = ContextEncoder("physiology", matrix).fit(train)
        if onehot.any() or len(physiology.fitted_tissues) != 10:
            raise RuntimeError(f"LOTO context contract failed: {path}")
        hashes[str(path.relative_to(STUDY_ROOT))] = sha256_file(path)
        summaries.append({
            "analysis_set": name,
            "heldout_tissue": heldout,
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "training_tissues": train["Tissue"].nunique(),
        })
    return data, hashes, summaries


def main() -> None:
    all_hashes = {}
    loto_summaries = []
    frames = {}
    for name, (data_path, split_root, expected_rows) in EXPECTED_DATASETS.items():
        frame, hashes, summaries = _validate_split_set(name, data_path, split_root, expected_rows)
        frames[name] = frame
        all_hashes.update(hashes)
        loto_summaries.extend(summaries)

    primary = frames["primary"]
    observed = (
        len(primary),
        primary["identity_unit_id"].nunique(),
        primary["parent_group_id"].nunique(),
        primary["SMILES"].nunique(),
        primary["Tissue"].nunique(),
    )
    if observed != (1270, 187, 134, 186, 11):
        raise RuntimeError(f"Primary identity contract failed: {observed}")

    manifests = build_job_manifests()
    expected_jobs = {"primary": 993, "ad_excluded": 762, "condition_specific": 762}
    if {name: len(frame) for name, frame in manifests.items()} != expected_jobs:
        raise RuntimeError("Job manifest count failed")
    cache_meta = json.loads(TABULAR_FEATURE_INDEX_PATH.read_text(encoding="utf-8"))
    if cache_meta["long_data_sha256"] != sha256_file(LONG_DATA_PATH):
        raise RuntimeError("Molecular feature cache checksum mismatch")

    forbidden = (
        "kp_" + "scripts",
        "results" + "/data",
        "results" + "/splits",
        "manuscript" + "/JPKPD2",
        "JPKPD2_" + "study/results",
    )
    dependency_hits = []
    for path in STUDY_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                dependency_hits.append({"path": str(path), "token": token})
    if dependency_hits:
        raise RuntimeError(f"Forbidden v1 dependencies: {dependency_hits}")

    preflight_path = STUDY_ROOT / "results" / "preflight" / "preflight_manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "passed"
        or preflight.get("retained_performance_values") is not False
        or len(preflight.get("checks", [])) != 15
    ):
        raise RuntimeError("15-path CPU integration preflight is not complete")
    gpu_preflight_path = STUDY_ROOT / "results" / "preflight" / "preflight_manifest_gpu.json"
    gpu_preflight = json.loads(gpu_preflight_path.read_text(encoding="utf-8"))
    if (
        gpu_preflight.get("status") != "passed"
        or gpu_preflight.get("retained_performance_values") is not False
        or gpu_preflight.get("device") != "gpu"
        or len(gpu_preflight.get("checks", [])) != 15
    ):
        raise RuntimeError("15-path GPU integration preflight is not complete")

    completed_full_jobs = (
        len(list(EXPERIMENT_RESULT_DIR.rglob("COMPLETE")))
        if EXPERIMENT_RESULT_DIR.exists()
        else 0
    )
    if completed_full_jobs:
        raise RuntimeError("Full v2 jobs already exist; pre-experiment freeze is no longer clean")

    code_files = [
        *sorted((STUDY_ROOT / "rat_kp_core").glob("*.py")),
        *(STUDY_ROOT / name for name in (
            "prepare_data.py",
            "prepare_experiment.py",
            "run_job.py",
            "run_experiments.py",
            "check_progress.py",
            "analyze_results.py",
            "run_preflight.py",
            "validate_setup.py",
        )),
    ]
    manifest_paths = {
        "primary": PRIMARY_JOB_MANIFEST_PATH,
        "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
        "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    }
    manifest = {
        "status": "passed",
        "scope": "v2 pre-experiment validation only; no full model job executed",
        "amendment_timing": "v1 outcomes observed; v2 identity rules frozen before any v2 model result",
        "data_contracts": {
            name: {
                "rows": len(frame),
                "identity_units": int(frame["identity_unit_id"].nunique()),
                "parent_groups": int(frame["parent_group_id"].nunique()),
                "model_smiles": int(frame["SMILES"].nunique()),
                "tissues": int(frame["Tissue"].nunique()),
                "sha256": sha256_file(EXPECTED_DATASETS[name][0]),
            }
            for name, frame in frames.items()
        },
        "split_files": len(all_hashes),
        "split_hashes": all_hashes,
        "loto_summaries": loto_summaries,
        "job_counts": expected_jobs,
        "job_manifest_hashes": {
            name: sha256_file(path) for name, path in manifest_paths.items()
        },
        "fixed_config_sha256": sha256_file(CONFIG_PATH),
        "tabular_feature_cache_sha256": sha256_file(TABULAR_FEATURE_CACHE_PATH),
        "forbidden_dependency_matches": 0,
        "preflight_checks": 15,
        "gpu_preflight_checks": 15,
        "completed_full_jobs": completed_full_jobs,
        "code_hashes": {
            str(path.relative_to(STUDY_ROOT)): sha256_file(path) for path in code_files
        },
        "statistical_lock": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260619,
            "loto_test": "two-sided exact sign-flip over 11 tissue deltas",
            "primary_fdr_family": "five model-specific primary-dataset LOTO physiology-vs-structure p-values only",
            "multiplicity": "Benjamini-Hochberg",
            "data_sensitivities": "exploratory robustness; excluded from primary FDR family",
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "chemprop": chemprop.__version__,
            "lightning": lightning.__version__,
            "rdkit": rdkit.__version__,
        },
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "pre_experiment_validation_v2.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "passed",
        "job_counts": expected_jobs,
        "completed_full_jobs": completed_full_jobs,
    }, indent=2))


if __name__ == "__main__":
    main()
