"""Neural architectures for the frozen fusion-position comparison.

The code calls the fusion position an "injection position"; see
``docs/terminology.md``. Every hyperparameter is read from the frozen
configuration rather than being passed in, so an architecture cannot be
instantiated with settings that differ from the recorded protocol.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP
from chemprop import nn as chemprop_nn

from rat_kp_core.models import StudyMPNN

from .config import load_config


def _head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Build the shared two-layer regression head used by all PyG models."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class BondWeightedGCNLayer(nn.Module):
    """GCN normalization with a learned positive scalar weight for every bond."""

    def __init__(self, hidden_dim: int, bond_dim: int):
        super().__init__()
        self.edge_gate = nn.Linear(bond_dim, 1)
        self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)

    def forward(self, x, edge_index, edge_attr):
        """Apply GCN propagation with a learned positive weight per bond."""
        edge_weight = F.softplus(self.edge_gate(edge_attr).reshape(-1)) + 1e-6
        return self.conv(x, edge_index, edge_weight=edge_weight)


class MolecularGCN(nn.Module):
    """Bond-weighted GCN encoder with early or late context fusion.

    Where context enters is the only thing that differs between conditions.

    Early fusion repeats the context vector across every atom and concatenates
    it to the atom features before the first message-passing step, so context
    can shape the learned representation. Late fusion concatenates the same
    vector to the pooled molecular representation just before the prediction
    head, so the representation is context-free and only the readout sees it.

    The context vector is used raw: it is never embedded or otherwise given
    learned parameters of its own, so the two positions differ in where the
    information enters rather than in how much capacity is spent on it.

    Under ``structure_only`` the context dimension is zero and both branches
    reduce to the plain architecture.
    """

    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int, injection_position: str):
        super().__init__()
        cfg = load_config()["gcn"]
        self.context_dim = int(context_dim)
        self.injection_position = injection_position
        early_dim = self.context_dim if injection_position == "early" else 0
        late_dim = self.context_dim if injection_position == "late" else 0
        self.atom_projection = nn.Linear(atom_dim + early_dim, cfg["hidden_dim"])
        self.layers = nn.ModuleList([
            BondWeightedGCNLayer(cfg["hidden_dim"], bond_dim)
            for _ in range(cfg["depth"])
        ])
        self.dropout = float(cfg["dropout"])
        self.head = _head(
            cfg["hidden_dim"] + late_dim,
            cfg["head_hidden_dim"],
            self.dropout,
        )

    def forward(self, batch):
        """Encode the molecule, fusing context at the configured position.

        Early fusion concatenates the context to every atom vector before message
        passing; late fusion concatenates it to the pooled representation before the
        head. Under ``structure_only`` the context width is zero and neither branch
        changes the input.
        """
        x = batch.x
        if self.injection_position == "early":
            x = torch.cat([x, batch.context[batch.batch]], dim=1)
        x = F.relu(self.atom_projection(x))
        for layer in self.layers:
            update = F.relu(layer(x, batch.edge_index, batch.edge_attr))
            x = x + F.dropout(update, p=self.dropout, training=self.training)
        graph = global_mean_pool(x, batch.batch)
        if self.injection_position == "late":
            graph = torch.cat([graph, batch.context], dim=1)
        return self.head(graph).reshape(-1)


class MolecularGINE(nn.Module):
    """GINE encoder with early or late context fusion.

    Residual updates and mean pooling match ``MolecularGCN`` so that the
    two architectures differ in their convolution alone. See
    ``MolecularGCN`` for how the two fusion positions work.
    """

    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int, injection_position: str):
        super().__init__()
        cfg = load_config()["gine"]
        self.context_dim = int(context_dim)
        self.injection_position = injection_position
        early_dim = self.context_dim if injection_position == "early" else 0
        late_dim = self.context_dim if injection_position == "late" else 0
        self.atom_projection = nn.Linear(atom_dim + early_dim, cfg["hidden_dim"])
        self.layers = nn.ModuleList()
        for _ in range(cfg["depth"]):
            update = nn.Sequential(
                nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
            )
            self.layers.append(GINEConv(
                update,
                eps=0.0,
                train_eps=cfg["train_eps"],
                edge_dim=bond_dim,
            ))
        self.dropout = float(cfg["dropout"])
        self.head = _head(
            cfg["hidden_dim"] + late_dim,
            cfg["head_hidden_dim"],
            self.dropout,
        )

    def forward(self, batch):
        """Encode the molecule, fusing context at the configured position.

        Early fusion concatenates the context to every atom vector before message
        passing; late fusion concatenates it to the pooled representation before the
        head. Under ``structure_only`` the context width is zero and neither branch
        changes the input.
        """
        x = batch.x
        if self.injection_position == "early":
            x = torch.cat([x, batch.context[batch.batch]], dim=1)
        x = F.relu(self.atom_projection(x))
        for layer in self.layers:
            update = F.relu(layer(x, batch.edge_index, batch.edge_attr))
            x = x + F.dropout(update, p=self.dropout, training=self.training)
        graph = global_mean_pool(x, batch.batch)
        if self.injection_position == "late":
            graph = torch.cat([graph, batch.context], dim=1)
        return self.head(graph).reshape(-1)


class InjectionAttentiveFP(nn.Module):
    """AttentiveFP encoder with early or late context fusion.

    Wraps the PyTorch Geometric AttentiveFP encoder, which performs its own
    attentive pooling, and applies the shared regression head. See
    ``MolecularGCN`` for how the two fusion positions work.
    """

    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int, injection_position: str):
        super().__init__()
        cfg = load_config()["attentive_fp"]
        self.context_dim = int(context_dim)
        self.injection_position = injection_position
        early_dim = self.context_dim if injection_position == "early" else 0
        late_dim = self.context_dim if injection_position == "late" else 0
        self.encoder = AttentiveFP(
            in_channels=atom_dim + early_dim,
            hidden_channels=cfg["hidden_channels"],
            out_channels=cfg["hidden_channels"],
            edge_dim=bond_dim,
            num_layers=cfg["num_layers"],
            num_timesteps=cfg["num_timesteps"],
            dropout=cfg["graph_dropout"],
        )
        self.head = _head(
            cfg["hidden_channels"] + late_dim,
            cfg["head_hidden_dim"],
            cfg["head_dropout"],
        )

    def forward(self, batch):
        """Encode the molecule with AttentiveFP, fusing context at the configured position."""
        x = batch.x
        if self.injection_position == "early":
            x = torch.cat([x, batch.context[batch.batch]], dim=1)
        graph = self.encoder(x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.injection_position == "late":
            graph = torch.cat([graph, batch.context], dim=1)
        return self.head(graph).reshape(-1)


def build_d_mpnn(context_dim: int, injection_position: str) -> StudyMPNN:
    """Build the D-MPNN with early or late context fusion.

    D-MPNN messages are initialized from the source atom's feature vector,
    so early fusion is expressed by widening the atom feature dimension:
    the context rides into every directed-bond initial message. Late fusion
    instead widens the predictor input. The atom width 72 is the Chemprop v2
    featurizer default recorded in ``docs/reproducibility_spec.md``.

    The learning-rate schedule is Chemprop's own warmup form, so it is given
    explicitly here while the weight decay is taken from the shared neural
    settings that every architecture uses.
    """
    cfg = load_config()["d_mpnn"]
    legacy = load_config()["shared_neural"]
    atom_dim = 72 + (context_dim if injection_position == "early" else 0)
    late_dim = context_dim if injection_position == "late" else 0
    message = chemprop_nn.BondMessagePassing(
        d_v=atom_dim,
        d_h=cfg["message_hidden_dim"],
        depth=cfg["message_passing_depth"],
        dropout=cfg["message_dropout"],
    )
    predictor = chemprop_nn.RegressionFFN(
        input_dim=message.output_dim + late_dim,
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
        weight_decay=legacy["weight_decay"],
    )


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable parameters, recorded per job to document capacity."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
