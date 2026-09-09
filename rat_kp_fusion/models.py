"""Neural architectures for the frozen injection-position comparison."""

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
        edge_weight = F.softplus(self.edge_gate(edge_attr).reshape(-1)) + 1e-6
        return self.conv(x, edge_index, edge_weight=edge_weight)


class MolecularGCN(nn.Module):
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
        x = batch.x
        if self.injection_position == "early":
            x = torch.cat([x, batch.context[batch.batch]], dim=1)
        graph = self.encoder(x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.injection_position == "late":
            graph = torch.cat([graph, batch.context], dim=1)
        return self.head(graph).reshape(-1)


def build_d_mpnn(context_dim: int, injection_position: str) -> StudyMPNN:
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
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
