"""Training loops: the arbitrarily trained teacher and the MASD student (Algorithm 1)."""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from .losses import masd_loss
from .memory import MemoryBank
from .metrics import accuracy
from .uncertainty import node_uncertainty


def _same_shapes(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa)


def train_supervised(model, data, epochs: int = 501, lr: float = 0.01, weight_decay: float = 1e-3,
                     patience: int | None = None, log_every: int = 100, verbose: bool = True):
    """Plain cross-entropy training on ``data.train_mask``.

    Used for the arbitrarily trained teacher T (notebooks: GAT, hidden 64, 8 heads,
    lr 0.01, weight decay 1e-3, 501 epochs, no early stopping) and for the
    "student from scratch" baseline. Returns ``(model, history)``; with
    ``patience`` set, the weights with the best validation accuracy are restored.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = -1.0, None, 0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        val_acc = accuracy(out, data.y, data.val_mask)
        test_acc = accuracy(out, data.y, data.test_mask)
        history.append({"epoch": epoch, "loss": loss.item(), "val_acc": val_acc, "test_acc": test_acc})
        if verbose and epoch % log_every == 0:
            print(f"[teacher] epoch {epoch:03d}  loss {loss.item():.4f}  val {val_acc:.4f}  test {test_acc:.4f}")

        if patience is not None:
            if val_acc > best_val:
                best_val, best_state, wait = val_acc, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"[teacher] trained in {time.time() - t0:.2f}s")
    return model, history


def train_masd(student, teacher, data, *, memory_policy: str = "divergent", epochs: int = 250,
               lr: float = 0.01, weight_decay: float = 7e-4, alpha: float = 1.0, beta: float = 0.005,
               gamma: float = 0.55, temperature: float = 10.0, uncertainty: str = "entropy",
               mc_passes: int = 10, loss_impl: str = "paper", memory_init: str = "teacher",
               patience: int | None = 50, select_on: str = "val", warmup: int = 0,
               seed: int = 0, log_every: int = 10, verbose: bool = True):
    """Memory Augmented Self-Distillation.

    Per epoch:
      1. forward the student, the frozen teacher and the memory snapshot;
      2. compute per-node uncertainty u(x) of the student (Eq. 6/7);
      3. L = α·CE + β·L_sim + γ·L_kd and an optimizer step (Eq. 9);
      4. let the memory policy decide whether S_{t-1} replaces S_m (Sect. 4.5).

    ``warmup`` epochs train with cross-entropy only before distillation starts
    (Sect. 5.1: "we can warm up the student network for some epochs").
    Model selection uses the validation mask by default; ``select_on='test'``
    mirrors the notebooks, which kept the best *test* accuracy.

    Returns ``(student, memory, history)`` with the best weights restored.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    init_state = None
    if memory_init == "teacher":
        if _same_shapes(student, teacher):  # paper: memory starts as the teacher's weights
            init_state = copy.deepcopy(teacher.state_dict())
        elif verbose:
            print("[masd] teacher and student architectures differ; memory starts empty")

    memory = MemoryBank(memory_policy, student, temperature=temperature, init_state=init_state, seed=seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    with torch.no_grad():
        teacher_lp = teacher(data.x, data.edge_index)

    best, best_state, wait, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(epochs):
        student.train()
        opt.zero_grad()
        student_lp = student(data.x, data.edge_index)

        if epoch < warmup:
            loss = F.nll_loss(student_lp[data.train_mask], data.y[data.train_mask])
            parts = {"ce": loss.item(), "sim": 0.0, "kd": 0.0}
        else:
            u = node_uncertainty(uncertainty, student, data.x, data.edge_index,
                                 student_lp.detach(), mc_passes=mc_passes)
            mem_lp = memory.log_probs(data.x, data.edge_index) if memory.ready else None
            loss, parts = masd_loss(student_lp, teacher_lp, mem_lp, data.y, data.train_mask, u,
                                    alpha, beta, gamma, temperature, impl=loss_impl)
        loss.backward()
        opt.step()
        memory.update(student, data.x, data.edge_index)

        student.eval()
        with torch.no_grad():
            out = student(data.x, data.edge_index)
        val_acc = accuracy(out, data.y, data.val_mask)
        test_acc = accuracy(out, data.y, data.test_mask)
        score = val_acc if select_on == "val" else test_acc
        history.append({"epoch": epoch, "loss": loss.item(), **parts, "val_acc": val_acc,
                        "test_acc": test_acc, "memory_replacements": memory.replacements})
        if verbose and epoch % log_every == 0:
            print(f"[masd:{memory_policy}] epoch {epoch:03d}  loss {loss.item():.4f}  "
                  f"(ce {parts['ce']:.3f} sim {parts['sim']:.4f} kd {parts['kd']:.4f})  "
                  f"val {val_acc:.4f}  test {test_acc:.4f}")

        if score > best:
            best, best_state, wait = score, copy.deepcopy(student.state_dict()), 0
        else:
            wait += 1
            if patience is not None and wait >= patience and epoch >= warmup:
                if verbose:
                    print(f"[masd] early stop at epoch {epoch} (no {select_on} gain for {patience} epochs)")
                break

    student.load_state_dict(best_state)
    if verbose:
        print(f"[masd] trained in {time.time() - t0:.2f}s, best {select_on} acc {best:.4f}, "
              f"memory replaced {memory.replacements}×")
    return student, memory, history
