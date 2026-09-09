"""Training engine for new injection-study jobs only."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from rat_kp_core.data import sha256_file
from rat_kp_core.models import attentive_graph
from rat_kp_core.physiology import ContextEncoder, load_physiology
from rat_kp_core.training import BestStateInMemory, load_job_frames

from .config import load_config
from .models import (
    InjectionAttentiveFP,
    MolecularGCN,
    MolecularGINE,
    build_d_mpnn,
    trainable_parameter_count,
)
from .paths import CONFIG_PATH, EXPERIMENT_RESULT_DIR, SMOKE_RESULT_DIR, STUDY_ROOT


PREDICTION_COLUMNS = [
    "job_id", "stage", "analysis_set", "model", "condition", "context_type",
    "injection_position", "split_type", "split_seed", "heldout_tissue",
    "training_seed", "row_index", "parent_group_id", "identity_unit_id",
    "Drug", "SMILES", "Tissue", "y_true", "y_pred",
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred, float) - np.asarray(y_true, float)
    absolute = np.abs(residual)
    return {
        "rmse_log10": float(np.sqrt(np.mean(residual ** 2))),
        "mae_log10": float(np.mean(absolute)),
        "within_two_fold": float(np.mean(absolute <= math.log10(2.0))),
        "within_three_fold": float(np.mean(absolute <= math.log10(3.0))),
    }


class GraphRegressor(pl.LightningModule):
    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network
        self.cfg = load_config()["shared_neural"]

    def forward(self, batch):
        return self.network(batch)

    def training_step(self, batch, batch_idx):
        loss = torch.nn.functional.mse_loss(self(batch), batch.y.reshape(-1))
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.nn.functional.mse_loss(self(batch), batch.y.reshape(-1))
        self.log("val_loss", loss, batch_size=batch.num_graphs)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch).detach()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )


def _trainer(device: str, smoke: bool):
    cfg = load_config()["shared_neural"]
    best = BestStateInMemory()
    callbacks = [best]
    if not smoke:
        from lightning.pytorch.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg["early_stopping_patience"],
            min_delta=cfg["early_stopping_min_delta"],
        ))
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=1 if smoke else cfg["max_epochs"],
        limit_train_batches=1 if smoke else 1.0,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        gradient_clip_val=cfg["gradient_clip_val"],
        log_every_n_steps=1,
    )
    return trainer, best


def _context(job: pd.Series, frames: dict[str, pd.DataFrame]):
    encoder = ContextEncoder(
        str(job["context_type"]),
        load_physiology(),
        scaling=str(job["physiology_scaling"]),
    ).fit(frames["train"])
    return {name: encoder.transform(frame).astype(np.float32) for name, frame in frames.items()}


def _pyg_graph(smiles: str, context: np.ndarray, target: float) -> Data:
    base = attentive_graph(smiles, torch.from_numpy(context), target)
    return base


def _pyg_datasets(frames, contexts):
    return {
        name: [
            _pyg_graph(row.SMILES, contexts[name][index], row.log10Kp)
            for index, row in enumerate(frame.itertuples(index=False))
        ]
        for name, frame in frames.items()
    }


def _pyg_loaders(graphs, seed: int):
    batch_size = load_config()["shared_neural"]["batch_size"]
    output = {}
    for name, values in graphs.items():
        generator = torch.Generator().manual_seed(seed)
        output[name] = PyGDataLoader(
            values,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
            generator=generator,
        )
    return output


def _fit_pyg(job, frames, contexts, device, smoke):
    graphs = _pyg_datasets(frames, contexts)
    loaders = _pyg_loaders(graphs, int(job["training_seed"]))
    first = graphs["train"][0]
    context_dim = contexts["train"].shape[1]
    position = str(job["injection_position"])
    if job["model"] == "gcn":
        network = MolecularGCN(first.x.shape[1], first.edge_attr.shape[1], context_dim, position)
    elif job["model"] == "gine":
        network = MolecularGINE(first.x.shape[1], first.edge_attr.shape[1], context_dim, position)
    elif job["model"] == "attentive_fp":
        network = InjectionAttentiveFP(first.x.shape[1], first.edge_attr.shape[1], context_dim, position)
    else:
        raise ValueError(f"Unsupported PyG model: {job['model']}")
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


def _chemprop_dataset(frame, context, injection_position: str):
    context_dim = context.shape[1]
    datapoints = []
    for index, row in enumerate(frame.itertuples(index=False)):
        molecule = Chem.MolFromSmiles(row.SMILES)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {row.SMILES}")
        kwargs = {}
        if injection_position == "early":
            kwargs["V_f"] = np.repeat(context[index][None, :], molecule.GetNumAtoms(), axis=0)
        elif injection_position == "late":
            kwargs["x_d"] = context[index]
        datapoints.append(MoleculeDatapoint(
            molecule,
            y=np.array([row.log10Kp], dtype=np.float32),
            **kwargs,
        ))
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=context_dim if injection_position == "early" else 0
    )
    return MoleculeDataset(datapoints, featurizer=featurizer)


def _fit_d_mpnn(job, frames, contexts, device, smoke):
    position = str(job["injection_position"])
    context_dim = contexts["train"].shape[1]
    datasets = {
        name: _chemprop_dataset(frame, contexts[name], position)
        for name, frame in frames.items()
    }
    batch_size = load_config()["shared_neural"]["batch_size"]
    loaders = {
        "train": build_dataloader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            seed=int(job["training_seed"]), num_workers=0,
        ),
        "val": build_dataloader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": build_dataloader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    model = build_d_mpnn(context_dim, position)
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


def execute_new_job(
    job: pd.Series,
    *,
    device: str = "auto",
    smoke: bool = False,
    output_root: Path | None = None,
) -> Path:
    if str(job["execution_mode"]) != "new":
        raise ValueError("Legacy reuse rows must never be trained by the extension runner")
    load_config(require_frozen=not smoke)
    root = output_root or (SMOKE_RESULT_DIR if smoke else EXPERIMENT_RESULT_DIR)
    output = root / str(job["analysis_set"]) / str(job["stage"]) / str(job["job_id"])
    complete_marker = output / ("SMOKE_COMPLETE" if smoke else "COMPLETE")
    if complete_marker.exists():
        if not (output / "run.json").exists() or not (output / "predictions.csv").exists():
            raise RuntimeError(f"Corrupt completed injection-study output: {output}")
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if previous.get("config_sha256") == sha256_file(CONFIG_PATH):
            return output
        if not smoke:
            raise RuntimeError(f"Completed output has a different config hash: {output}")
        complete_marker.unlink()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pl.seed_everything(int(job["training_seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    frames = load_job_frames(job)
    contexts = _context(job, frames)
    if job["model"] == "d_mpnn":
        prediction, metadata = _fit_d_mpnn(job, frames, contexts, device, smoke)
    else:
        prediction, metadata = _fit_pyg(job, frames, contexts, device, smoke)
    if len(prediction) != len(frames["test"]) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction output")
    predictions = frames["test"][[
        "row_index", "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue"
    ]].copy()
    predictions["y_true"] = frames["test"]["log10Kp"].to_numpy(float)
    predictions["y_pred"] = prediction
    for column in PREDICTION_COLUMNS[:11]:
        predictions[column] = job[column]
    predictions = predictions[PREDICTION_COLUMNS]
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    run = {
        "status": "smoke_complete" if smoke else "complete",
        "job": job.to_dict(),
        "metrics": metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()),
        "training_metadata": metadata,
        "dataset_sha256": sha256_file(STUDY_ROOT / str(job["dataset_path"])),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runtime_seconds": time.time() - started,
        "smoke": smoke,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    complete_marker.write_text("complete\n", encoding="utf-8")
    failed = output / "FAILED.json"
    if failed.exists():
        failed.unlink()
    return output
