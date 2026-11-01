"""Per-node prediction uncertainty in [0, 1] (paper Eq. 6 and Sect. 6.4).

Shannon entropy is the paper's default. Monte Carlo dropout, margin and least
confidence are the alternatives compared in Fig. 6; MC dropout is the variant
used in the GAT notebook cells.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Normalised Shannon entropy: H(p) / log C, so ``U`` in Eq. 7 is ``log C``."""
    p = log_probs.exp()
    h = -(p * torch.log(p + 1e-6)).sum(-1)
    return h / math.log(p.size(-1))


@torch.no_grad()
def margin(log_probs: torch.Tensor) -> torch.Tensor:
    """1 − (p_top1 − p_top2)."""
    top2 = log_probs.exp().topk(2, dim=-1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


@torch.no_grad()
def least_confidence(log_probs: torch.Tensor) -> torch.Tensor:
    """(1 − max p) rescaled to [0, 1] by its maximum value 1 − 1/C."""
    c = log_probs.size(-1)
    return (1.0 - log_probs.exp().max(-1).values) / (1.0 - 1.0 / c)


@torch.no_grad()
def mc_dropout(model, x, edge_index, passes: int = 10) -> torch.Tensor:
    """Mean predictive variance over ``passes`` stochastic forward passes,
    normalised by its maximum (as in the notebook's ``mc_dropout_uncertainty``)."""
    was_training = model.training
    model.train()
    probs = torch.stack([model(x, edge_index).exp() for _ in range(passes)])  # [T, N, C]
    model.train(was_training)
    u = probs.var(dim=0).mean(dim=1)
    return u / u.max().clamp_min(1e-12)


def node_uncertainty(kind: str, model, x, edge_index, log_probs, mc_passes: int = 10):
    kind = kind.lower()
    if kind == "entropy":
        return entropy(log_probs)
    if kind == "margin":
        return margin(log_probs)
    if kind in ("least_confidence", "lc"):
        return least_confidence(log_probs)
    if kind in ("mc_dropout", "mc"):
        return mc_dropout(model, x, edge_index, passes=mc_passes)
    raise ValueError(f"Unknown uncertainty '{kind}'. Use entropy | mc_dropout | margin | least_confidence.")
