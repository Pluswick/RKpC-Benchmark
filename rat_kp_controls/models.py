"""Additive molecule-plus-tissue-intercept controls for all four GNN families."""

from __future__ import annotations

import torch
from torch import nn
from chemprop import nn as chemprop_nn

from rat_kp_core.models import StudyMPNN
from rat_kp_fusion.config import load_config as load_base_config
from rat_kp_fusion.models import (
    InjectionAttentiveFP,
    MolecularGCN,
    MolecularGINE,
)


class CenteredTissueIntercept(nn.Module):
    """Per-tissue offsets constrained to sum to zero.

    Centering removes the redundancy between the offsets and the prediction
    head's own bias, which would otherwise be free to trade against each
    other and leave the individual offsets unidentifiable.
    """

    def __init__(self, n_tissues: int = 11):
        super().__init__()
        self.raw_effect = nn.Parameter(torch.zeros(n_tissues))

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """Return the sum-to-zero tissue offset selected by each row's one-hot vector."""
        effect = self.raw_effect - self.raw_effect.mean()
        return onehot @ effect


class AdditivePyGRegressor(nn.Module):
    """Molecule-only neural predictor plus a scalar tissue intercept.

    The control the manuscript calls molecule + additive tissue intercept.
    The encoder never sees context, so the only way tissue can affect a
    prediction is an additive per-tissue shift. It therefore isolates how
    much of the tissue-context benefit is a pure location shift, which a
    full-context model must beat to show it uses context for anything more.

    Both terms are fitted jointly on training rows.
    """

    def __init__(self, molecular_network: nn.Module, n_tissues: int = 11):
        super().__init__()
        self.molecular_network = molecular_network
        self.tissue_intercept = CenteredTissueIntercept(n_tissues)

    def forward(self, batch):
        """Add the tissue offset to a molecule-only prediction."""
        molecular_prediction = self.molecular_network(batch)
        return molecular_prediction + self.tissue_intercept(batch.context)


def build_additive_pyg(model_key: str, atom_dim: int, bond_dim: int) -> AdditivePyGRegressor:
    """Build one PyG architecture as a molecule-only encoder plus tissue intercept.

    The encoder is constructed with zero context dimension and fusion position
    ``none``, so it is exactly the structure-only architecture and the two
    conditions differ only by the added intercept.
    """
    builders = {
        "gcn": MolecularGCN,
        "gine": MolecularGINE,
        "attentive_fp": InjectionAttentiveFP,
    }
    if model_key not in builders:
        raise ValueError(f"Unsupported additive PyG model: {model_key}")
    molecular = builders[model_key](atom_dim, bond_dim, 0, "none")
    return AdditivePyGRegressor(molecular, n_tissues=11)


class AdditiveRegressionFFN(chemprop_nn.RegressionFFN):
    """Chemprop molecular FFN plus a centered scalar tissue intercept.

    Chemprop passes one concatenated feature matrix to the predictor, so the
    molecular block and the tissue one-hot are split apart here by width: the
    FFN sees only the molecular columns and the intercept only the tissue
    columns. That keeps the D-MPNN control additive in the same sense as the
    PyG one, where the split is structural.
    """

    def __init__(self, *, molecular_dim: int, context_dim: int, **kwargs):
        super().__init__(input_dim=molecular_dim, **kwargs)
        self.molecular_dim = int(molecular_dim)
        self.context_dim = int(context_dim)
        self.tissue_intercept = CenteredTissueIntercept(context_dim)
        self.hparams["molecular_dim"] = self.molecular_dim
        self.hparams["context_dim"] = self.context_dim

    @property
    def input_dim(self) -> int:
        """Total input width Chemprop sees: molecular block plus tissue one-hot."""
        return self.molecular_dim + self.context_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Split the feature matrix by width, then add the tissue offset to the FFN output."""
        molecular = self.output_transform(self.ffn(features[:, :self.molecular_dim]))
        tissue = self.tissue_intercept(features[:, self.molecular_dim:]).reshape(-1, 1)
        return molecular + tissue

    train_step = forward


def build_additive_d_mpnn() -> StudyMPNN:
    """Build the D-MPNN additive control, sharing the frozen D-MPNN settings."""
    cfg = load_base_config()["d_mpnn"]
    shared = load_base_config()["shared_neural"]
    message = chemprop_nn.BondMessagePassing(
        d_v=72,
        d_h=cfg["message_hidden_dim"],
        depth=cfg["message_passing_depth"],
        dropout=cfg["message_dropout"],
    )
    predictor = AdditiveRegressionFFN(
        molecular_dim=message.output_dim,
        context_dim=11,
        hidden_dim=cfg["head_hidden_dim"],
        n_layers=cfg["head_num_layers"],
        dropout=cfg["head_dropout"],
        criterion=chemprop_nn.MSE(),
    )
    return StudyMPNN(
        message,
        chemprop_nn.MeanAggregation(),
        predictor,
        batch_norm=False,
        metrics=[chemprop_nn.RMSE(), chemprop_nn.MAE()],
        warmup_epochs=2,
        init_lr=0.0001,
        max_lr=0.001,
        final_lr=0.0001,
        weight_decay=shared["weight_decay"],
    )
