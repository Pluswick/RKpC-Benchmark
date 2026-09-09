from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop.data import build_dataloader

from rat_kp_core.training import load_job_frames
from rat_kp_core.training import _chemprop_dataset as _legacy_chemprop_dataset
from rat_kp_core.training import _trainer as _legacy_trainer
from rat_kp_core.models import build_d_mpnn as build_legacy_d_mpnn
from rat_kp_core.models import load_config as load_legacy_config
from rat_kp_core.physiology import ContextEncoder, load_physiology
from rat_kp_fusion.config import load_config as load_base_config
from rat_kp_fusion.models import MolecularGINE, build_d_mpnn, trainable_parameter_count
from rat_kp_fusion.paths import EXPERIMENT_RESULT_DIR as INJECTION_RESULT_DIR, STUDY_ROOT
from rat_kp_fusion.training import (
    GraphRegressor, _chemprop_dataset, _pyg_datasets, _pyg_loaders, _trainer,
)
from .config import load_config
from .paths import EXPERIMENT_DIR


def _full_grid(test: pd.DataFrame) -> pd.DataFrame:
    tissues = list(load_config()["tissue_volumes_ml"])
    identities = test[["parent_group_id", "identity_unit_id", "Drug", "SMILES"]].drop_duplicates()
    rows = []
    for identity in identities.itertuples(index=False):
        for tissue in tissues:
            rows.append({
                "row_index": f"grid::{identity.identity_unit_id}::{tissue}",
                "parent_group_id": identity.parent_group_id,
                "identity_unit_id": identity.identity_unit_id,
                "Drug": identity.Drug,
                "SMILES": identity.SMILES,
                "Tissue": tissue,
                "log10Kp": 0.0,
            })
    return pd.DataFrame(rows)


def _canonical_path(job: pd.Series) -> Path:
    if str(job["execution_mode"]) == "legacy_reuse":
        return STUDY_ROOT / str(job["legacy_result_path"]) / "predictions.csv"
    return INJECTION_RESULT_DIR / str(job["analysis_set"]) / str(job["stage"]) / str(job["job_id"]) / "predictions.csv"


def _fit_predict(job: pd.Series, frames: dict[str, pd.DataFrame], contexts: dict[str, np.ndarray], device: str):
    if str(job["model"]) == "gine":
        graphs = _pyg_datasets(frames, contexts)
        loaders = _pyg_loaders(graphs, int(job["training_seed"]))
        first = graphs["train"][0]
        network = MolecularGINE(
            first.x.shape[1], first.edge_attr.shape[1], contexts["train"].shape[1],
            str(job["injection_position"]),
        )
        model = GraphRegressor(network)
        parameter_count = trainable_parameter_count(network)
    elif str(job["model"]) == "d_mpnn" and str(job["execution_mode"]) == "legacy_reuse":
        datasets = {name: _legacy_chemprop_dataset(frame, contexts[name]) for name, frame in frames.items()}
        cfg = load_legacy_config()["d_mpnn"]
        loaders = {
            "train": build_dataloader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True, seed=int(job["training_seed"]), num_workers=0),
            "val": build_dataloader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
            "test": build_dataloader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
            "grid": build_dataloader(datasets["grid"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        }
        model = build_legacy_d_mpnn(contexts["train"].shape[1])
        parameter_count = trainable_parameter_count(model)
        trainer, best = _legacy_trainer(cfg, device, smoke=False)
        trainer.fit(model, loaders["train"], loaders["val"])
        best.restore(model)
        result = {}
        for name in ("test", "grid"):
            batches = trainer.predict(model, dataloaders=loaders[name])
            result[name] = torch.cat([value.detach().cpu().reshape(-1) for value in batches]).numpy()
        return result, {
            "trainable_parameters": parameter_count,
            "best_val_loss": best.best_score,
            "best_epoch_zero_based": best.best_epoch,
            "epochs_completed": int(trainer.current_epoch),
            "legacy_architecture_reused": True,
        }
    elif str(job["model"]) == "d_mpnn":
        position = str(job["injection_position"])
        datasets = {name: _chemprop_dataset(frame, contexts[name], position) for name, frame in frames.items()}
        batch_size = load_base_config()["shared_neural"]["batch_size"]
        loaders = {
            "train": build_dataloader(datasets["train"], batch_size=batch_size, shuffle=True, seed=int(job["training_seed"]), num_workers=0),
            "val": build_dataloader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
            "test": build_dataloader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
            "grid": build_dataloader(datasets["grid"], batch_size=batch_size, shuffle=False, num_workers=0),
        }
        model = build_d_mpnn(contexts["train"].shape[1], position)
        parameter_count = trainable_parameter_count(model)
    else:
        raise ValueError(f"Unsupported representative model: {job['model']}")
    trainer, best = _trainer(device, smoke=False)
    trainer.fit(model, loaders["train"], loaders["val"])
    best.restore(model)
    result = {}
    for name in ("test", "grid"):
        batches = trainer.predict(model, dataloaders=loaders[name])
        result[name] = torch.cat([value.detach().cpu().reshape(-1) for value in batches]).numpy()
    return result, {
        "trainable_parameters": parameter_count,
        "best_val_loss": best.best_score,
        "best_epoch_zero_based": best.best_epoch,
        "epochs_completed": int(trainer.current_epoch),
    }


def execute_job(job: pd.Series, *, device: str = "gpu") -> Path:
    load_base_config(require_frozen=True)
    output = EXPERIMENT_DIR / str(job["split_type"]) / str(job["job_id"])
    marker = output / "COMPLETE"
    if marker.exists():
        return output
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pl.seed_everything(int(job["training_seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    base_frames = load_job_frames(job)
    frames = dict(base_frames)
    frames["grid"] = _full_grid(base_frames["test"])
    encoder = ContextEncoder(
        str(job["context_type"]), load_physiology(), scaling=str(job["physiology_scaling"])
    ).fit(frames["train"])
    contexts = {name: encoder.transform(frame).astype(np.float32) for name, frame in frames.items()}
    predicted, metadata = _fit_predict(job, frames, contexts, device)
    observed = frames["test"][["row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue", "log10Kp"]].copy()
    observed = observed.rename(columns={"log10Kp": "y_true"})
    observed["y_pred"] = predicted["test"]
    grid = frames["grid"][["row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue"]].copy()
    grid["y_pred"] = predicted["grid"]
    for frame in (observed, grid):
        frame.insert(0, "condition", str(job["condition"]))
        frame.insert(0, "model", str(job["model"]))
        frame.insert(0, "split_seed", int(job["split_seed"]))
        frame.insert(0, "split_type", str(job["split_type"]))
    canonical = pd.read_csv(_canonical_path(job), encoding="utf-8-sig")
    check = observed.merge(canonical[["row_index", "y_pred"]], on="row_index", suffixes=("_new", "_canonical"), validate="one_to_one")
    max_abs_difference = float(np.max(np.abs(check["y_pred_new"] - check["y_pred_canonical"])))
    if max_abs_difference > 1e-4:
        (output / "REPRODUCIBILITY_FAILURE.json").write_text(json.dumps({
            "max_abs_difference_log10": max_abs_difference,
            "new_rmse_log10": float(np.sqrt(np.mean((observed["y_pred"] - observed["y_true"]) ** 2))),
            "canonical_rmse_log10": float(np.sqrt(np.mean((canonical["y_pred"] - canonical["y_true"]) ** 2))),
            "training_metadata": metadata,
            "execution_mode": str(job["execution_mode"]),
        }, indent=2), encoding="utf-8")
        raise RuntimeError(f"Retraining reproducibility check failed: max |delta|={max_abs_difference:.6g}")
    observed.to_csv(output / "observed_predictions.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(output / "full_grid_predictions.csv", index=False, encoding="utf-8-sig")
    run = {
        "status": "complete",
        "job_id": str(job["job_id"]),
        "split_type": str(job["split_type"]),
        "split_seed": int(job["split_seed"]),
        "model": str(job["model"]),
        "condition": str(job["condition"]),
        "canonical_prediction_path": str(_canonical_path(job)),
        "max_abs_retraining_difference_log10": max_abs_difference,
        "n_test_rows": len(observed),
        "n_grid_rows": len(grid),
        "training_metadata": metadata,
        "runtime_seconds": time.time() - started,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    marker.write_text("complete\n", encoding="utf-8")
    return output
