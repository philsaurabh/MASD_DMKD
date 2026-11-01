"""Evaluation metrics reported in the paper (Sect. 5.4, Table 5): top-1 accuracy,
macro precision / recall / F1, expected calibration error (ECE), linear CKA
between student and teacher embeddings, and GFLOPs via fvcore."""
from __future__ import annotations

import math
import warnings

import torch
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------- classification
@torch.no_grad()
def classification_report(log_probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
    pred = log_probs.argmax(-1)[mask].cpu().numpy()
    true = y[mask].cpu().numpy()
    return {
        "accuracy": float((pred == true).mean()),
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(true, pred, average="macro", zero_division=0),
    }


@torch.no_grad()
def accuracy(log_probs, y, mask) -> float:
    return (log_probs.argmax(-1)[mask] == y[mask]).float().mean().item()


@torch.no_grad()
def expected_calibration_error(log_probs, y, mask, num_classes: int, n_bins: int = 15) -> float:
    """Top-label ECE on the masked nodes (torchmetrics, L1 norm)."""
    from torchmetrics.classification import MulticlassCalibrationError
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm="l1")
    return ece(log_probs[mask].exp().cpu(), y[mask].cpu()).item()


# ---------------------------------------------------------------- CKA
class CKA:
    """Centered Kernel Alignment (Kornblith et al., 2019) in PyTorch, runs on the
    tensors' device. Adapted from github.com/yuanli2333/CKA-Centered-Kernel-Alignment."""

    @staticmethod
    def _center(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
        return H @ K @ H

    @classmethod
    def linear_hsic(cls, X, Y):
        return (cls._center(X @ X.T) * cls._center(Y @ Y.T)).sum()

    @classmethod
    def linear(cls, X, Y) -> float:
        X, Y = X.double(), Y.double()
        return (cls.linear_hsic(X, Y) / torch.sqrt(cls.linear_hsic(X, X) * cls.linear_hsic(Y, Y))).item()

    @staticmethod
    def _rbf(X, sigma=None):
        G = X @ X.T
        D = torch.diag(G)[:, None] - 2 * G + torch.diag(G)[None, :]
        if sigma is None:
            sigma = math.sqrt(torch.median(D[D != 0]).item())
        return torch.exp(-0.5 * D / sigma ** 2)

    @classmethod
    def kernel(cls, X, Y, sigma=None) -> float:
        X, Y = X.double(), Y.double()
        kx, ky = cls._center(cls._rbf(X, sigma)), cls._center(cls._rbf(Y, sigma))
        return ((kx * ky).sum() / torch.sqrt((kx * kx).sum() * (ky * ky).sum())).item()


@torch.no_grad()
def cka_similarity(student, teacher, x, edge_index, max_nodes: int = 5000, seed: int = 0) -> float:
    """Linear CKA between hidden embeddings. CKA builds ``N×N`` Gram matrices, so
    large graphs are sub-sampled to ``max_nodes`` nodes."""
    es, et = student.get_embeddings(x, edge_index), teacher.get_embeddings(x, edge_index)
    if es.size(0) > max_nodes:
        idx = torch.randperm(es.size(0), generator=torch.Generator().manual_seed(seed))[:max_nodes]
        es, et = es[idx.to(es.device)], et[idx.to(et.device)]
    return CKA.linear(es, et)


# ---------------------------------------------------------------- FLOPs
def count_gflops(model, x, edge_index) -> float | None:
    """Forward-pass GFLOPs with fvcore. fvcore skips scatter/gather operators it
    does not know (most message passing), so treat the number as a lower bound."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    was_training = model.training
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops = FlopCountAnalysis(model, (x, edge_index))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    model.train(was_training)
    return total / 1e9

