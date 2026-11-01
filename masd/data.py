"""Dataset loading and subgraph sampling.

Node-classification benchmarks used in the paper (Sect. 5.2): Cora, Citeseer,
Pubmed (Planetoid public splits) and Amazon Photo / Computers (random per-class
splits, since PyG ships them without masks).

Following Sect. 4.2 and the notebooks, the student is trained on a subgraph
drawn with ``LinkNeighborLoader``: edges are sampled first, then a fixed number
of neighbours is expanded around their end-points.
"""
from __future__ import annotations

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.loader import LinkNeighborLoader

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"photo": "Photo", "computers": "Computers"}


def random_class_split(data: Data, num_classes: int, train_per_class: int = 20,
                       val_per_class: int = 30, seed: int = 0) -> Data:
    """Per-class random split (Shchur et al., 2018): ``train_per_class`` and
    ``val_per_class`` labelled nodes per class, everything else is test."""
    g = torch.Generator().manual_seed(seed)
    n = data.num_nodes
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        train[idx[:train_per_class]] = True
        val[idx[train_per_class:train_per_class + val_per_class]] = True
    data.train_mask, data.val_mask = train, val
    data.test_mask = ~(train | val)
    return data


def load_dataset(name: str, root: str = "data", split_seed: int = 0):
    """Return ``(data, num_features, num_classes)`` for a node-classification dataset."""
    key = name.lower()
    if key in PLANETOID:
        ds = Planetoid(root=f"{root}/{PLANETOID[key]}", name=PLANETOID[key],
                       transform=T.NormalizeFeatures())
        data = ds[0]
    elif key in AMAZON:
        ds = Amazon(root=f"{root}/Amazon{AMAZON[key]}", name=AMAZON[key],
                    transform=T.NormalizeFeatures())
        data = random_class_split(ds[0], ds.num_classes, seed=split_seed)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from "
                         f"{sorted(list(PLANETOID) + list(AMAZON))}.")
    return data, ds.num_features, ds.num_classes


def sample_subgraph(data: Data, num_neighbors: int = 100, num_hops: int = 10,
                    batch_size: int = 512, seed: int = 0) -> Data:
    """Draw one training subgraph with ``LinkNeighborLoader`` (as in the notebooks:
    ``num_neighbors=[100] * 10``, ``batch_size=512``). Node-level attributes,
    including the train/val/test masks, are carried over to the subgraph."""
    torch.manual_seed(seed)
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[num_neighbors] * num_hops,
        batch_size=batch_size,
        edge_label_index=data.edge_index,
        shuffle=True,
    )
    return next(iter(loader))
