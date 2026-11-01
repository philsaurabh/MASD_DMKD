"""GNN backbones used as teacher and student (paper Sect. 5.3).

Every model shares one interface so the training code is architecture-agnostic:

* ``forward(x, edge_index)`` returns **log-probabilities** ``[N, C]``;
* ``get_embeddings(x, edge_index)`` returns the last hidden representation,
  used for CKA similarity.

Layer configurations follow the notebooks. ``GCN`` and ``GraphSAGE`` are not
defined in the notebooks (the GCN cells call an undefined ``GCN(hidden_channels=...)``),
so they use the standard two-layer PyG formulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv


class _Base(nn.Module):
    def _hidden(self, x, edge_index):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _head(self, h, edge_index):  # pragma: no cover
        raise NotImplementedError

    def forward(self, x, edge_index, return_embeddings: bool = False):
        h = self._hidden(x, edge_index)
        if return_embeddings:
            return h
        return F.log_softmax(self._head(h, edge_index), dim=-1)

    @torch.no_grad()
    def get_embeddings(self, x, edge_index):
        was_training = self.training
        self.eval()
        h = self.forward(x, edge_index, return_embeddings=True)
        self.train(was_training)
        return h


class GAT(_Base):
    """Two-layer GAT (notebook default: hidden 64, 8 heads, dropout 0.5)."""

    def __init__(self, in_dim, num_classes, hidden=64, heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def _hidden(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self.conv1(x, edge_index))

    def _head(self, h, edge_index):
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GAT3(_Base):
    """Three-layer GAT (``GAT1`` in the notebooks)."""

    def __init__(self, in_dim, num_classes, hidden=64, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def _hidden(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return F.elu(self.conv2(x, edge_index))

    def _head(self, h, edge_index):
        return self.conv3(h, edge_index)


class GCN(_Base):
    """Two-layer GCN (Kipf & Welling, 2017)."""

    def __init__(self, in_dim, num_classes, hidden=64, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def _hidden(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.conv1(x, edge_index))

    def _head(self, h, edge_index):
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GraphSAGE(_Base):
    """Two-layer GraphSAGE (Hamilton et al., 2017) with mean aggregation."""

    def __init__(self, in_dim, num_classes, hidden=64, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, num_classes)

    def _hidden(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.conv1(x, edge_index))

    def _head(self, h, edge_index):
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class MLP(_Base):
    """Non-message-passing baseline ("Graph MLP"): ignores ``edge_index``."""

    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc1, self.bn1 = nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden)
        self.fc2, self.bn2 = nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def _hidden(self, x, edge_index=None):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), p=self.dropout, training=self.training)
        return F.relu(self.bn2(self.fc2(x)))

    def _head(self, h, edge_index=None):
        return self.fc3(F.dropout(h, p=self.dropout, training=self.training))


class GraphTransformer(_Base):
    """Three-layer graph transformer (Yun et al., 2019) with batch norm."""

    def __init__(self, in_dim, num_classes, hidden=512, heads=4, dropout=0.5, attn_dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=attn_dropout)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.conv2 = TransformerConv(hidden * heads, hidden, heads=heads, dropout=attn_dropout)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.conv3 = TransformerConv(hidden * heads, num_classes, heads=1, dropout=attn_dropout)

    def _hidden(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.bn2(self.conv2(x, edge_index)))

    def _head(self, h, edge_index):
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv3(h, edge_index)


MODELS = {
    "gat": GAT,
    "gat3": GAT3,
    "gcn": GCN,
    "sage": GraphSAGE,
    "mlp": MLP,
    "transformer": GraphTransformer,
}


def build_model(name: str, in_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a backbone by name. ``kwargs`` override the defaults above;
    keys a model does not accept are ignored, so one CLI serves all models."""
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}.")
    cls = MODELS[key]
    accepted = cls.__init__.__code__.co_varnames
    return cls(in_dim, num_classes, **{k: v for k, v in kwargs.items() if k in accepted and v is not None})
