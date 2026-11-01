"""MASD training objective (paper Eq. 9):  L = α·L_ce + β·L_sim + γ·L_kd.

Two implementations of the distillation terms are provided:

``paper``     follows Sect. 4.4 exactly:
    L_sim = MSE between the teacher's and student's tempered soft predictions
            on training nodes (Eq. 2);
    L_kd  = mean_x [ ψ1(x)·KL(S_t||S_m) + ψ2(x)·KL(S_t||T) ] · τ²     (Eqs. 3–5)
    with ψ1 = u(x), ψ2 = 1 − u(x) from normalised uncertainty (Eq. 7), so the
    student leans on memory when it is uncertain and on the teacher otherwise.

``notebook``  reproduces the loss as written in the research notebooks:
    both KL terms are computed with ``F.kl_div`` on probabilities (not
    log-probabilities), compare *student vs teacher* and *memory vs teacher*,
    are divided by τ², and are weighted by the same uncertainty u(x).
    Note that the memory term does not depend on the student's parameters, so
    in this mode memory affects the loss value but not the gradients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def tempered_log_probs(log_probs: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.log_softmax(log_probs / tau, dim=-1)


def kl_per_node(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for every node, from log-probabilities (gradients flow through p)."""
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def masd_loss(student_lp, teacher_lp, memory_lp, y, train_mask, uncertainty,
              alpha: float, beta: float, gamma: float, tau: float, impl: str = "paper"):
    """Return ``(total, parts)``.

    Args:
        student_lp: student log-probabilities ``[N, C]`` (with grad).
        teacher_lp: teacher log-probabilities ``[N, C]`` (no grad).
        memory_lp:  memory log-probabilities at temperature τ, or ``None`` if
                    memory is empty (the student's own detached prediction is used,
                    as in the notebooks).
        uncertainty: per-node uncertainty in [0, 1], ``[N]`` (no grad).
    """
    s_t = tempered_log_probs(student_lp, tau)
    t_t = tempered_log_probs(teacher_lp, tau)
    m_t = memory_lp if memory_lp is not None else s_t.detach()

    loss_ce = F.nll_loss(student_lp[train_mask], y[train_mask])
    loss_sim = F.mse_loss(s_t[train_mask].exp(), t_t[train_mask].exp())

    if impl == "paper":
        psi1, psi2 = uncertainty, 1.0 - uncertainty
        kl_mem = kl_per_node(s_t, m_t)
        kl_teacher = kl_per_node(s_t, t_t)
        loss_kd = (psi1 * kl_mem + psi2 * kl_teacher).mean() * tau ** 2
    elif impl == "notebook":
        s_p, t_p, m_p = s_t.exp(), t_t.exp(), m_t.exp()
        b1 = F.kl_div(s_p, t_p, reduction="none").sum(-1) / tau ** 2
        b2 = F.kl_div(m_p, t_p, reduction="none").sum(-1) / tau ** 2
        loss_kd = (b1 * uncertainty).mean() + (b2 * uncertainty).mean()
    else:
        raise ValueError("impl must be 'paper' or 'notebook'")

    total = alpha * loss_ce + beta * loss_sim + gamma * loss_kd
    return total, {"ce": loss_ce.item(), "sim": loss_sim.item(), "kd": loss_kd.item()}
