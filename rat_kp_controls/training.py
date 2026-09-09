"""Training engine for the isolated additional context analyses."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop.data import build_dataloader

from rat_kp_core.data import sha256_file
from rat_kp_core.physiology import ContextEncoder, PHYSIOLOGY_COLUMNS, load_physiology
from rat_kp_core.training import load_job_frames
from rat_kp_fusion.config import load_config as load_base_config
from rat_kp_fusion.models import trainable_parameter_count
from rat_kp_fusion.paths import CONFIG_PATH as BASE_CONFIG_PATH
from rat_kp_fusion.training import (
    GraphRegressor,
    _chemprop_dataset,
    _fit_d_mpnn,
    _fit_pyg,
    _pyg_datasets,
    _pyg_loaders,
    _trainer,
    metrics,
)

from .config import load_config
from .models import build_additive_d_mpnn, build_additive_pyg
from .paths import CONFIG_PATH, EXPERIMENT_RESULT_DIR, SMOKE_RESULT_DIR, STUDY_ROOT


PREDICTION_COLUMNS = [
    "job_id", "analysis", "stage", "analysis_set", "model", "condition",
    "context_type", "injection_position", "physiology_scaling", "split_type",
    "split_seed", "heldout_tissue", "training_seed", "row_index",
    "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue",
    "y_true", "y_pred",
]


def _contexts(job: pd.Series, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    if str(job["condition"]) == "additive_tissue_intercept":
        encoder = ContextEncoder("tissue_onehot", load_physiology()).fit(frames["train"])
        if len(encoder.fitted_tissues) != 11:
            raise RuntimeError("Primary additive control requires all 11 tissues in training")
        return {
            name: encoder.transform(frame).astype(np.float32)
            for name, frame in frames.items()
        }
    if str(job["physiology_scaling"]) != "raw_fraction":
        raise ValueError("Unknown additional-analysis context preprocessing")
    matrix = load_physiology().set_index("tissue")[PHYSIOLOGY_COLUMNS]
    output = {}
    for name, frame in frames.items():
        tissues = frame["Tissue"].astype(str).str.lower().tolist()
        values = matrix.loc[tissues].to_numpy(np.float32)
        # The first four descriptors are fractions. Convert the sole mg/g
        # descriptor to g/g so the sensitivity input is unit-harmonized.
        values[:, 4] /= 1000.0
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError("Unit-harmonized physiology inputs must be finite and within [0, 1]")
        output[name] = values
    return output


def _fit_additive_pyg(job, frames, contexts, device, smoke):
    graphs = _pyg_datasets(frames, contexts)
    loaders = _pyg_loaders(graphs, int(job["training_seed"]))
    first = graphs["train"][0]
    network = build_additive_pyg(
        str(job["model"]), first.x.shape[1], first.edge_attr.shape[1]
    )
    parameter_count = trainable_parameter_count(network)
    model = GraphRegressor(network)
    trainer, best = _trainer(device, smoke)
    trainer.fit(model, loaders["train"], loaders["val"])
    best.restore(model)
    batches = trainer.predict(model, dataloaders=loaders["test"])
    prediction = torch.cat([value.cpu().reshape(-1) for value in batches]).numpy()
    return prediction, {
        "trainable_parameters": parameter_count,
        "best_val_loss": best.best_score,
        "best_epoch_zero_based": best.best_epoch,
        "epochs_completed": int(trainer.current_epoch),
    }


def _fit_additive_d_mpnn(job, frames, contexts, device, smoke):
    datasets = {
        name: _chemprop_dataset(frame, contexts[name], "late")
        for name, frame in frames.items()
    }
    batch_size = load_base_config()["shared_neural"]["batch_size"]
    loaders = {
        "train": build_dataloader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            seed=int(job["training_seed"]), num_workers=0,
        ),
        "val": build_dataloader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
        "test": build_dataloader(
            datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    model = build_additive_d_mpnn()
    parameter_count = trainable_parameter_count(model)
    trainer, best = _trainer(device, smoke)
    trainer.fit(model, loaders["train"], loaders["val"])
    best.restore(model)
    batches = trainer.predict(model, dataloaders=loaders["test"])
    prediction = torch.cat([value.detach().cpu().reshape(-1) for value in batches]).numpy()
    return prediction, {
        "trainable_parameters": parameter_count,
        "best_val_loss": best.best_score,
        "best_epoch_zero_based": best.best_epoch,
        "epochs_completed": int(trainer.current_epoch),
    }


def _serializable(value):
    if pd.isna(value):
        return ""
    return value.item() if hasattr(value, "item") else value


def execute_job(
    job: pd.Series,
    *,
    device: str = "auto",
    smoke: bool = False,
    output_root: Path | None = None,
) -> Path:
    load_base_config(require_frozen=True)
    load_config(require_locked=not smoke)
    root = output_root or (SMOKE_RESULT_DIR if smoke else EXPERIMENT_RESULT_DIR)
    output = root / str(job["analysis"]) / str(job["job_id"])
    marker = output / ("SMOKE_COMPLETE" if smoke else "COMPLETE")
    current_hash = sha256_file(CONFIG_PATH)
    if marker.exists():
        run_path = output / "run.json"
        prediction_path = output / "predictions.csv"
        if not run_path.exists() or not prediction_path.exists():
            raise RuntimeError(f"Corrupt completed additional-analysis output: {output}")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("additional_config_sha256") == current_hash:
            return output
        if not smoke:
            raise RuntimeError(f"Completed output has a different protocol hash: {output}")
        marker.unlink()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pl.seed_everything(int(job["training_seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    frames = load_job_frames(job)
    contexts = _contexts(job, frames)
    if str(job["condition"]) == "additive_tissue_intercept":
        if str(job["model"]) == "d_mpnn":
            prediction, metadata = _fit_additive_d_mpnn(job, frames, contexts, device, smoke)
        else:
            prediction, metadata = _fit_additive_pyg(job, frames, contexts, device, smoke)
    elif str(job["model"]) == "d_mpnn":
        prediction, metadata = _fit_d_mpnn(job, frames, contexts, device, smoke)
    else:
        prediction, metadata = _fit_pyg(job, frames, contexts, device, smoke)
    if len(prediction) != len(frames["test"]) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid additional-analysis prediction output")
    predictions = frames["test"][[
        "row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue"
    ]].copy()
    predictions["y_true"] = frames["test"]["log10Kp"].to_numpy(float)
    predictions["y_pred"] = prediction
    for column in PREDICTION_COLUMNS[:13]:
        predictions[column] = job[column]
    predictions = predictions[PREDICTION_COLUMNS]
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    run = {
        "status": "smoke_complete" if smoke else "complete",
        "job": {key: _serializable(value) for key, value in job.to_dict().items()},
        "metrics": metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()),
        "training_metadata": metadata,
        "dataset_sha256": sha256_file(STUDY_ROOT / str(job["dataset_path"])),
        "split_sha256": sha256_file(STUDY_ROOT / str(job["split_path"])),
        "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
        "additional_config_sha256": current_hash,
        "runtime_seconds": time.time() - started,
        "smoke": smoke,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    marker.write_text("complete\n", encoding="utf-8")
    failed = output / "FAILED.json"
    if failed.exists():
        failed.unlink()
    return output
