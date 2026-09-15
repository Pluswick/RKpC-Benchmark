"""Single-job training and prediction engine for all five frozen model families."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
import warnings

from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

from .data import read_csv, sha256_file
from .features import TabularPreprocessor
from .models import (
    AttentiveFPLightning,
    attentive_graph,
    build_d_mpnn,
    build_tabular,
    load_config,
)
from .paths import CONFIG_PATH, EXPERIMENT_RESULT_DIR, LONG_DATA_PATH, STUDY_ROOT
from .physiology import ContextEncoder, load_physiology


PREDICTION_COLUMNS = [
    "job_id", "stage", "analysis_set", "model", "context", "physiology_scaling",
    "split_type", "split_seed", "heldout_tissue", "training_seed", "row_index",
    "parent_group_id", "identity_unit_id", "Drug", "SMILES", "Tissue", "y_true", "y_pred",
]


def load_job_frames(job: pd.Series) -> dict[str, pd.DataFrame]:
    """Load and partition one job's dataset, re-verifying every leakage guarantee.

    The split file carries a copy of the identity columns and they are checked
    against the dataset after the join, so a split built for a different dataset
    revision cannot be applied silently.

    Leakage checks are re-run here at training time rather than trusted from
    split construction: no parent group may span train and validation, none may
    reach the observed-tissue test partition, and under LOTO the held-out tissue
    must be absent from training entirely. A violation raises rather than
    producing an optimistic result.
    """
    dataset_path = STUDY_ROOT / str(job["dataset_path"])
    long_df = read_csv(dataset_path)
    required_data = {"row_index", "parent_group_id", "identity_unit_id", "SMILES", "Tissue", "log10Kp"}
    if required_data - set(long_df.columns):
        raise ValueError(f"Dataset missing columns: {sorted(required_data - set(long_df.columns))}")
    split_path = STUDY_ROOT / str(job["split_path"])
    split = read_csv(split_path)
    expected = {"row_index", "parent_group_id", "identity_unit_id", "SMILES", "Tissue", "split"}
    if expected - set(split.columns):
        raise ValueError(f"Split missing columns: {sorted(expected - set(split.columns))}")
    merged = long_df.merge(
        split[list(expected)],
        on="row_index",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    identity_columns = ("parent_group_id", "identity_unit_id", "SMILES", "Tissue")
    if not all((merged[column] == merged[f"{column}_split"]).all() for column in identity_columns):
        raise ValueError("Split identity mismatch")
    merged = merged.drop(columns=[f"{column}_split" for column in identity_columns])
    frames = {
        label: merged[merged["split"] == label].reset_index(drop=True)
        for label in ("train", "val", "test")
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError(f"Empty train/val/test partition for {job['job_id']}")
    if set(frames["train"]["parent_group_id"]) & set(frames["val"]["parent_group_id"]):
        raise ValueError("Parent-compound leakage between train and validation")
    if job["stage"] == "part1" and (
        (set(frames["train"]["parent_group_id"]) | set(frames["val"]["parent_group_id"]))
        & set(frames["test"]["parent_group_id"])
    ):
        raise ValueError("Parent-compound leakage into Part 1 test")
    if job["stage"] == "loto":
        heldout = str(job["heldout_tissue"])
        if set(frames["test"]["Tissue"]) != {heldout}:
            raise ValueError("LOTO test tissue mismatch")
        if heldout in set(frames["train"]["Tissue"]) | set(frames["val"]["Tissue"]):
            raise ValueError("Held-out tissue leaked into train/validation")
    return frames


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Test metrics on log10(Kp), including 2- and 3-fold accuracy fractions."""
    residual = y_pred - y_true
    abs_residual = np.abs(residual)
    return {
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(abs_residual)),
        "within_two_fold": float(np.mean(abs_residual <= math.log10(2.0))),
        "within_three_fold": float(np.mean(abs_residual <= math.log10(3.0))),
    }


def _context_arrays(job: pd.Series, frames: dict[str, pd.DataFrame]):
    """Fit the context encoder on training rows and encode every partition."""
    encoder = ContextEncoder(
        str(job["context"]),
        load_physiology(),
        scaling=str(job["physiology_scaling"]),
    ).fit(frames["train"])
    return encoder, {key: encoder.transform(frame) for key, frame in frames.items()}


def _fit_tabular(job: pd.Series, frames: dict[str, pd.DataFrame], smoke: bool):
    """Fit one tabular baseline and predict the test partition.

    Preprocessing statistics are fit on training rows only. Fit-time warnings
    are captured and stored in the run record rather than printed, so a
    convergence warning stays attached to the job that produced it.
    """
    encoder = ContextEncoder(
        str(job["context"]),
        load_physiology(),
        scaling=str(job["physiology_scaling"]),
    )
    prep = TabularPreprocessor(str(job["model"]), encoder)
    x_train = prep.fit_transform(frames["train"])
    x_val = prep.transform(frames["val"])
    x_test = prep.transform(frames["test"])
    y_train = frames["train"]["log10Kp"].to_numpy(float)
    y_val = frames["val"]["log10Kp"].to_numpy(float)
    model = build_tabular(str(job["model"]), int(job["training_seed"]), smoke=smoke)
    caught = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        if job["model"] == "xgboost":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        else:
            model.fit(x_train, y_train)
        caught = [str(record.message) for record in records]
    metadata = {
        "feature_count": int(x_train.shape[1]),
        "warnings": caught,
        "best_iteration": (
            int(model.best_iteration)
            if hasattr(model, "best_iteration") and model.best_iteration is not None
            else None
        ),
        "validation_metrics": _metrics(y_val, np.asarray(model.predict(x_val), float)),
    }
    return np.asarray(model.predict(x_test), float), metadata


def _chemprop_dataset(frame: pd.DataFrame, context: np.ndarray) -> MoleculeDataset:
    """Build a Chemprop dataset with context as a molecule-level feature."""
    return MoleculeDataset([
        MoleculeDatapoint.from_smi(
            row.SMILES,
            y=np.array([row.log10Kp], dtype=np.float32),
            x_d=np.asarray(context[index], dtype=np.float32),
        )
        for index, row in enumerate(frame.itertuples(index=False))
    ])


def _trainer(
    cfg: dict,
    device: str,
    smoke: bool,
) -> tuple[pl.Trainer, "BestStateInMemory"]:
    """Build the Lightning trainer and the in-memory best-state callback.

    Deterministic execution is required and checkpointing is disabled; the best
    validation state is kept in memory and restored before prediction instead.
    ``smoke`` reduces the run to a single epoch on a single batch and drops
    early stopping.
    """
    best_state = BestStateInMemory()
    callbacks = [best_state]
    if not smoke:
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
    return trainer, best_state


class BestStateInMemory(Callback):
    """Retain the lowest-validation-loss weights without checkpoint deserialization."""

    def __init__(self):
        super().__init__()
        self.best_score = float("inf")
        self.best_epoch = None
        self.best_state = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """Capture the weights whenever validation loss reaches a new minimum.

        Sanity-check passes are ignored, since they run before training and would
        otherwise set the baseline from an untrained model.
        """
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get("val_loss")
        if score is None:
            return
        value = float(score.detach().cpu())
        if value < self.best_score:
            self.best_score = value
            self.best_epoch = int(trainer.current_epoch)
            self.best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in pl_module.state_dict().items()
            }

    def restore(self, model):
        """Load the retained best weights, raising if none were ever captured."""
        if self.best_state is None:
            raise RuntimeError("No validation state was captured")
        model.load_state_dict(self.best_state)


def _fit_d_mpnn(job, frames, context, output_dir, device, smoke):
    """Train the core-stage D-MPNN and predict the test partition."""
    cfg = load_config()["d_mpnn"]
    datasets = {key: _chemprop_dataset(frames[key], context[key]) for key in frames}
    loaders = {
        "train": build_dataloader(
            datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
            seed=int(job["training_seed"]), num_workers=0,
        ),
        "val": build_dataloader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        "test": build_dataloader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
    }
    model = build_d_mpnn(context["train"].shape[1])
    trainer, best_state = _trainer(cfg, device, smoke)
    trainer.fit(model, loaders["train"], loaders["val"])
    best_state.restore(model)
    batches = trainer.predict(model, dataloaders=loaders["test"])
    prediction = torch.cat([batch.detach().cpu().reshape(-1) for batch in batches]).numpy()
    metadata = {
        "best_val_loss": best_state.best_score,
        "best_epoch_zero_based": best_state.best_epoch,
        "epochs_completed": int(trainer.current_epoch),
    }
    return prediction, metadata


def _pyg_loader(graphs, batch_size, shuffle, seed):
    """Build a PyG loader with a seeded generator and no worker processes."""
    generator = torch.Generator().manual_seed(seed)
    return PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator
    )


def _fit_attentive(job, frames, context, output_dir, device, smoke):
    """Train the core-stage AttentiveFP and predict the test partition."""
    cfg = load_config()["attentive_fp"]
    graphs = {
        key: [
            attentive_graph(row.SMILES, torch.from_numpy(context[key][index]), row.log10Kp)
            for index, row in enumerate(frame.itertuples(index=False))
        ]
        for key, frame in frames.items()
    }
    loaders = {
        key: _pyg_loader(
            graph_list, cfg["batch_size"], key == "train", int(job["training_seed"])
        )
        for key, graph_list in graphs.items()
    }
    first = graphs["train"][0]
    model = AttentiveFPLightning(first.x.shape[1], first.edge_attr.shape[1], context["train"].shape[1])
    trainer, best_state = _trainer(cfg, device, smoke)
    trainer.fit(model, loaders["train"], loaders["val"])
    best_state.restore(model)
    batches = trainer.predict(model, dataloaders=loaders["test"])
    prediction = torch.cat([batch.detach().cpu().reshape(-1) for batch in batches]).numpy()
    metadata = {
        "best_val_loss": best_state.best_score,
        "best_epoch_zero_based": best_state.best_epoch,
        "epochs_completed": int(trainer.current_epoch),
    }
    return prediction, metadata


def _prediction_frame(job: pd.Series, test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Assemble the frozen prediction schema for one job.

    Every row keeps its job, split, seed, source row, parent group, identity
    unit, SMILES, and tissue alongside the observed and predicted target, so a
    prediction file can be re-aggregated later without consulting the plan.
    Prediction count and finiteness are asserted before anything is written.
    """
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError("Prediction count or finiteness check failed")
    result = pd.DataFrame({
        "row_index": test["row_index"].astype(int),
        "parent_group_id": test["parent_group_id"].astype(str),
        "identity_unit_id": test["identity_unit_id"].astype(str),
        "Drug": test["Drug"].astype(str),
        "SMILES": test["SMILES"].astype(str),
        "Tissue": test["Tissue"].astype(str),
        "y_true": test["log10Kp"].astype(float),
        "y_pred": prediction.astype(float),
    })
    for column in PREDICTION_COLUMNS[:10]:
        result[column] = job[column] if column in job else ""
    return result[PREDICTION_COLUMNS]


def execute_job(
    job: pd.Series,
    *,
    device: str = "auto",
    smoke: bool = False,
    output_root: Path = EXPERIMENT_RESULT_DIR,
) -> Path:
    """Execute exactly one manifest row and write a completion-marked result directory."""
    output_dir = output_root / str(job["analysis_set"]) / str(job["stage"]) / str(job["job_id"])
    complete = output_dir / "COMPLETE"
    if complete.exists() and not smoke:
        if (output_dir / "run.json").exists() and (output_dir / "predictions.csv").exists():
            return output_dir
        raise RuntimeError(f"Corrupt completed job directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pl.seed_everything(int(job["training_seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    frames = load_job_frames(job)
    try:
        if job["model"] in {"random_forest", "xgboost", "linear_svr"}:
            prediction, training_metadata = _fit_tabular(job, frames, smoke)
        else:
            _, context = _context_arrays(job, frames)
            if job["model"] == "d_mpnn":
                prediction, training_metadata = _fit_d_mpnn(
                    job, frames, context, output_dir, device, smoke
                )
            elif job["model"] == "attentive_fp":
                prediction, training_metadata = _fit_attentive(
                    job, frames, context, output_dir, device, smoke
                )
            else:
                raise ValueError(f"Unknown model: {job['model']}")
        predictions = _prediction_frame(job, frames["test"], prediction)
        predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        run = {
            "status": "complete",
            "job": {
                key: (
                    "" if pd.isna(value)
                    else value.item() if hasattr(value, "item")
                    else value
                )
                for key, value in job.to_dict().items()
            },
            "test_metrics": _metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()),
            "training_metadata": training_metadata,
            "partition_rows": {key: len(value) for key, value in frames.items()},
            "partition_parent_groups": {key: int(value["parent_group_id"].nunique()) for key, value in frames.items()},
            "partition_identity_units": {key: int(value["identity_unit_id"].nunique()) for key, value in frames.items()},
            "dataset_sha256": sha256_file(STUDY_ROOT / str(job["dataset_path"])),
            "fixed_config_sha256": sha256_file(CONFIG_PATH),
            "runtime_seconds": time.time() - started,
            "smoke_override": smoke,
        }
        (output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        complete.write_text("complete\n", encoding="utf-8")
        failed_path = output_dir / "FAILED.json"
        if failed_path.exists():
            failed_path.unlink()
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return output_dir
    except Exception as exc:
        (output_dir / "FAILED.json").write_text(
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise
