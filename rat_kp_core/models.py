"""Self-contained frozen model builders for the independent study."""

from __future__ import annotations

import json

from lightning import pytorch as pl
import torch
from torch import nn as torch_nn
from chemprop import nn as chemprop_nn
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn.models import AttentiveFP

from .data import sha256_file
from .paths import CONFIG_PATH


CONFIG_SHA256 = "08d56f8a43bb4ab0c1b551f099e60dbe5b3f7b9ca13c73833f4c670954113c4f"


def load_config() -> dict:
    """Load the frozen core (v2) model configuration, refusing any unrecorded edit.

    Pinned by checksum and re-checked for the frozen status marker, so an
    edited protocol cannot produce results labelled as frozen.
    """
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("Independent fixed-model config checksum mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "approved_and_frozen_before_v2_execution" or config.get("target") != "log10(Kp)":
        raise ValueError("Fixed config contract failed")
    return config


def build_tabular(model_key: str, seed: int, smoke: bool = False):
    """Build one tabular baseline with its frozen hyperparameters.

    These three baselines belong to the core stage and are not reported in the
    manuscript; see ``docs/terminology.md``.

    ``random_state_policy`` in the configuration records that the seed is
    supplied per run rather than fixed in the file, so it is replaced here by
    the actual run seed. LinearSVR is deterministic under the frozen settings
    and carries a fixed seed instead.

    ``smoke`` shrinks the model so an integration check runs quickly; smoke
    metrics are never reported.
    """
    cfg = dict(load_config()[model_key])
    if model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        cfg.pop("random_state_policy")
        cfg["random_state"] = seed
        if smoke:
            cfg.update(n_estimators=3, n_jobs=1)
        return RandomForestRegressor(**cfg)
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        cfg.pop("random_state_policy")
        cfg["random_state"] = seed
        if smoke:
            cfg.update(n_estimators=3, early_stopping_rounds=2, n_jobs=1)
        return XGBRegressor(**cfg)
    if model_key == "linear_svr":
        from sklearn.svm import LinearSVR

        if smoke:
            cfg["max_iter"] = 100
        return LinearSVR(**cfg)
    raise ValueError(f"Unknown tabular model: {model_key}")


class StudyMPNN(MPNN):
    """Chemprop MPNN with the frozen weight decay and no external project wrapper."""

    def __init__(self, *args, weight_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.study_weight_decay = float(weight_decay)

    def configure_optimizers(self):
        """Apply the frozen weight decay on top of Chemprop's own optimizer.

        Chemprop builds the optimizer and its warmup schedule internally, so the
        weight decay is set on the returned parameter groups rather than by
        constructing a different optimizer, which would discard that schedule.
        """
        result = super().configure_optimizers()
        for group in result["optimizer"].param_groups:
            group["weight_decay"] = self.study_weight_decay
        return result


def build_d_mpnn(context_dim: int) -> StudyMPNN:
    """Build the core-stage D-MPNN, widening the predictor input by the context.

    The core stage fuses context only at the predictor, so there is no fusion
    position argument here; the fusion-position comparison lives in
    ``rat_kp_fusion.models``.
    """
    cfg = load_config()["d_mpnn"]
    message = chemprop_nn.BondMessagePassing(
        d_h=cfg["message_hidden_dim"],
        depth=cfg["message_passing_depth"],
        dropout=cfg["message_dropout"],
    )
    predictor = chemprop_nn.RegressionFFN(
        input_dim=message.output_dim + context_dim,
        hidden_dim=cfg["ffn_hidden_dim"],
        n_layers=cfg["ffn_num_layers"],
        dropout=cfg["ffn_dropout"],
        criterion=chemprop_nn.MSE(),
    )
    return StudyMPNN(
        message,
        chemprop_nn.MeanAggregation(),
        predictor,
        batch_norm=cfg["batch_norm"],
        metrics=[chemprop_nn.RMSE(), chemprop_nn.MAE()],
        warmup_epochs=cfg["warmup_epochs"],
        init_lr=cfg["init_lr"],
        max_lr=cfg["max_lr"],
        final_lr=cfg["final_lr"],
        weight_decay=cfg["weight_decay"],
    )


class AttentiveFPWithContext(torch_nn.Module):
    """AttentiveFP encoder whose pooled representation is concatenated with context."""

    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int):
        super().__init__()
        cfg = load_config()["attentive_fp"]
        self.encoder = AttentiveFP(
            in_channels=atom_dim,
            hidden_channels=cfg["hidden_channels"],
            out_channels=cfg["hidden_channels"],
            edge_dim=bond_dim,
            num_layers=cfg["num_layers"],
            num_timesteps=cfg["num_timesteps"],
            dropout=cfg["graph_dropout"],
        )
        self.head = torch_nn.Sequential(
            torch_nn.Linear(cfg["hidden_channels"] + context_dim, cfg["ffn_hidden_dim"]),
            torch_nn.ReLU(),
            torch_nn.Dropout(cfg["ffn_dropout"]),
            torch_nn.Linear(cfg["ffn_hidden_dim"], 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, context):
        """Encode the graph, then concatenate context before the prediction head."""
        graph = self.encoder(x, edge_index, edge_attr, batch)
        return self.head(torch.cat([graph, context], dim=1)).view(-1)


class AttentiveFPLightning(pl.LightningModule):
    """Frozen AttentiveFP training wrapper with validation-only early stopping support."""

    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int):
        super().__init__()
        self.save_hyperparameters()
        self.network = AttentiveFPWithContext(atom_dim, bond_dim, context_dim)
        self.config = load_config()["attentive_fp"]

    def forward(self, batch):
        """Unpack a PyG batch into the wrapped network's positional arguments."""
        return self.network(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.context
        )

    def training_step(self, batch, batch_idx):
        """Mean squared error on log10(Kp) for one training batch."""
        loss = torch.nn.functional.mse_loss(self(batch), batch.y.reshape(-1))
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation loss, the quantity early stopping and best-state selection watch."""
        loss = torch.nn.functional.mse_loss(self(batch), batch.y.reshape(-1))
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Detached predictions for one batch."""
        return self(batch).detach()

    def configure_optimizers(self):
        """AdamW with the frozen learning rate and weight decay."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )


def attentive_graph(smiles: str, context: torch.Tensor, target: float) -> Data:
    """Featurize one molecule into a PyG graph carrying its context and target.

    Uses the Chemprop v2 featurizer defaults so the PyG architectures and the
    D-MPNN see identical atom and bond descriptions, leaving the architecture
    as the only difference between them.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    graph = SimpleMoleculeMolGraphFeaturizer()(molecule)
    return Data(
        x=torch.as_tensor(graph.V, dtype=torch.float32),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(graph.E, dtype=torch.float32),
        context=context.float().reshape(1, -1),
        y=torch.tensor([target], dtype=torch.float32),
    )
