"""Write and validate the injection-study manifests without running full experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform

import chemprop
import lightning
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
import torch_geometric

from rat_kp_core.data import sha256_file
from rat_kp_core.paths import STUDY_ROOT

from rat_kp_fusion.config import CONFIG_SHA256, load_config
from rat_kp_fusion.jobs import write_plan
from rat_kp_fusion.models import (
    InjectionAttentiveFP,
    MolecularGCN,
    MolecularGINE,
    build_d_mpnn,
    trainable_parameter_count,
)
from rat_kp_fusion.paths import (
    CONFIG_PATH,
    EXPERIMENT_RESULT_DIR,
    GPU_SMOKE_RESULT_DIR,
    MANIFEST_DIR,
    NEW_JOB_MANIFEST_PATH,
    PLAN_MANIFEST_PATH,
    PREFLIGHT_REPORT_PATH,
    REUSE_MANIFEST_PATH,
    SMOKE_RESULT_DIR,
)
from rat_kp_fusion.training import _pyg_graph


def _reuse_audit(reuse: pd.DataFrame) -> dict:
    config = load_config(require_frozen=True)
    expected_legacy_hash = config["legacy_fixed_config_sha256"]
    prediction_rows = 0
    for row in reuse.itertuples(index=False):
        root = STUDY_ROOT / row.legacy_result_path
        required = [root / "COMPLETE", root / "run.json", root / "predictions.csv"]
        if not all(path.exists() for path in required) or (root / "FAILED.json").exists():
            raise RuntimeError(f"Invalid legacy reuse output: {row.legacy_job_id}")
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if run["job"]["job_id"] != row.legacy_job_id:
            raise RuntimeError(f"Legacy metadata mismatch: {row.legacy_job_id}")
        if run["fixed_config_sha256"] != expected_legacy_hash:
            raise RuntimeError(f"Legacy config mismatch: {row.legacy_job_id}")
        predictions = pd.read_csv(root / "predictions.csv", encoding="utf-8-sig")
        if len(predictions) != int(run["partition_rows"]["test"]):
            raise RuntimeError(f"Legacy prediction count mismatch: {row.legacy_job_id}")
        if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all():
            raise RuntimeError(f"Legacy non-finite predictions: {row.legacy_job_id}")
        prediction_rows += len(predictions)
    return {"jobs": len(reuse), "prediction_rows": prediction_rows, "status": "passed"}


def _split_audit(plan: pd.DataFrame) -> dict:
    unique = plan[["split_path", "stage", "heldout_tissue"]].drop_duplicates()
    leakage = 0
    for row in unique.itertuples(index=False):
        split = pd.read_csv(STUDY_ROOT / row.split_path, encoding="utf-8-sig")
        if row.stage == "part1":
            counts = split.groupby("parent_group_id")["split"].nunique()
        else:
            counts = split[split["split"] != "test"].groupby("parent_group_id")["split"].nunique()
            if set(split.loc[split["split"] == "test", "Tissue"]) != {row.heldout_tissue}:
                raise RuntimeError(f"LOTO tissue mismatch: {row.split_path}")
        leakage += int((counts > 1).sum())
    if leakage:
        raise RuntimeError(f"Parent-group leakage detected: {leakage}")
    return {"unique_split_files": int(unique["split_path"].nunique()), "leakage_count": leakage}


def _parameter_counts() -> list[dict]:
    rows = []
    conditions = {
        "structure_only": (0, "none"),
        "onehot_late": (11, "late"),
        "onehot_early": (11, "early"),
        "physiology_late": (5, "late"),
        "physiology_early": (5, "early"),
    }
    base = _pyg_graph("CCO", np.empty(0, dtype=np.float32), 0.0)
    for condition, (context_dim, position) in conditions.items():
        builders = {
            "gcn": lambda: MolecularGCN(base.x.shape[1], base.edge_attr.shape[1], context_dim, position),
            "gine": lambda: MolecularGINE(base.x.shape[1], base.edge_attr.shape[1], context_dim, position),
            "d_mpnn": lambda: build_d_mpnn(context_dim, position),
            "attentive_fp": lambda: InjectionAttentiveFP(
                base.x.shape[1], base.edge_attr.shape[1], context_dim, position
            ),
        }
        for model, builder in builders.items():
            rows.append({
                "model": model,
                "condition": condition,
                "trainable_parameters": trainable_parameter_count(builder()),
            })
    return rows


def _smoke_audit(config_hash: str) -> dict:
    runs = list(SMOKE_RESULT_DIR.rglob("run.json"))
    if len(runs) != 14:
        raise RuntimeError(f"Expected 14 smoke runs, found {len(runs)}")
    models = set()
    conditions = set()
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "smoke_complete" or run.get("config_sha256") != config_hash:
            raise RuntimeError(f"Invalid smoke run: {path}")
        predictions = pd.read_csv(path.parent / "predictions.csv", encoding="utf-8-sig")
        if predictions.empty or not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all():
            raise RuntimeError(f"Invalid smoke predictions: {path}")
        models.add(run["job"]["model"])
        conditions.add(run["job"]["condition"])
    return {
        "jobs": len(runs),
        "models": sorted(models),
        "conditions_exercised": sorted(conditions),
        "status": "passed",
    }


def _gpu_smoke_audit(config_hash: str) -> dict:
    runs = list(GPU_SMOKE_RESULT_DIR.rglob("run.json"))
    if len(runs) != 4:
        raise RuntimeError(f"Expected 4 GPU smoke runs, found {len(runs)}")
    models = set()
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "smoke_complete" or run.get("config_sha256") != config_hash:
            raise RuntimeError(f"Invalid GPU smoke run: {path}")
        predictions = pd.read_csv(path.parent / "predictions.csv", encoding="utf-8-sig")
        if predictions.empty or not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all():
            raise RuntimeError(f"Invalid GPU smoke predictions: {path}")
        models.add(run["job"]["model"])
    expected = {"gcn", "gine", "d_mpnn", "attentive_fp"}
    if models != expected:
        raise RuntimeError(f"GPU smoke model coverage mismatch: {models}")
    return {"jobs": len(runs), "models": sorted(models), "status": "passed"}


def main() -> None:
    if CONFIG_SHA256 is None:
        raise RuntimeError("Freeze CONFIG_SHA256 before final preflight")
    config = load_config(require_frozen=True)
    plan, new, reuse = write_plan()
    if EXPERIMENT_RESULT_DIR.exists() and any(EXPERIMENT_RESULT_DIR.rglob("COMPLETE")):
        raise RuntimeError("Full injection experiments have already started")
    reuse_report = _reuse_audit(reuse)
    split_report = _split_audit(plan)
    config_hash = sha256_file(CONFIG_PATH)
    smoke_report = _smoke_audit(config_hash)
    gpu_smoke_report = _gpu_smoke_audit(config_hash)
    source_files = sorted((STUDY_ROOT / "rat_kp_fusion").glob("*.py"))
    source_files += [
        STUDY_ROOT / "prepare_injection_study.py",
        STUDY_ROOT / "run_injection_smoke.py",
        STUDY_ROOT / "run_injection_gpu_preflight.py",
        STUDY_ROOT / "run_injection_experiments.py",
        STUDY_ROOT / "analyze_injection_results.py",
        STUDY_ROOT / "tests" / "test_injection_contract.py",
        STUDY_ROOT / "manuscript" / "INJECTION_STUDY_PROTOCOL_V1_KO.md",
    ]
    report = {
        "status": "passed",
        "scope": "pre-experiment validation only; no full experiment launched",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "legacy_config_sha256": config["legacy_fixed_config_sha256"],
        "jobs": {
            "planned": len(plan),
            "new": len(new),
            "legacy_reuse": len(reuse),
            "primary_part1": int(((plan.analysis_set == "primary") & (plan.stage == "part1")).sum()),
            "primary_loto": int(((plan.analysis_set == "primary") & (plan.stage == "loto")).sum()),
            "ad_excluded_part1": int((plan.analysis_set == "ad_excluded").sum()),
            "condition_specific_part1": int((plan.analysis_set == "condition_specific").sum()),
        },
        "manifest_sha256": {
            "plan": sha256_file(PLAN_MANIFEST_PATH),
            "new": sha256_file(NEW_JOB_MANIFEST_PATH),
            "reuse": sha256_file(REUSE_MANIFEST_PATH),
        },
        "dataset_hashes": {
            relative: sha256_file(STUDY_ROOT / relative)
            for relative in sorted(plan["dataset_path"].unique())
        },
        "split_hashes": {
            relative: sha256_file(STUDY_ROOT / relative)
            for relative in sorted(plan["split_path"].unique())
        },
        "reuse_audit": reuse_report,
        "split_audit": split_report,
        "smoke_audit": smoke_report,
        "gpu_smoke_audit": gpu_smoke_report,
        "parameter_counts": _parameter_counts(),
        "full_experiments_started": False,
        "source_hashes": {
            str(path.relative_to(STUDY_ROOT)): sha256_file(path)
            for path in source_files
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "chemprop": chemprop.__version__,
            "lightning": lightning.__version__,
            "rdkit": rdkit.__version__,
        },
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "planned_jobs": len(plan),
        "new_jobs": len(new),
        "legacy_reuse_jobs": len(reuse),
        "full_experiments_started": False,
        "report": str(PREFLIGHT_REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
