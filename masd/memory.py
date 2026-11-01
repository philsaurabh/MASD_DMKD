"""Memory of past student snapshots and its replacement policies (paper Sect. 4.3, 4.5).

The memory holds exactly one past state of the student (weights only), so its
cost is one model copy regardless of the number of epochs. After each epoch the
policy decides whether the snapshot from the previous epoch, ``S_{t-1}``,
replaces the stored memory ``S_m``:

* ``1sb``        — 1-step back: always replace (memory = previous epoch).
* ``divergent``  — replace if KL(S_t || S_{t-1}) > KL(S_t || S_m), i.e. keep the
  snapshot whose distribution is *further* from the current student.
* ``convergent`` — replace if KL(S_t || S_{t-1}) < KL(S_t || S_m), i.e. keep the
  *closer* snapshot.
* ``random``     — memory is a uniformly random past epoch, maintained with
  reservoir sampling (replace with probability 1/t), so no history is stored.

KL(a || b) is computed between temperature-scaled softmax outputs, averaged
over nodes.
"""
from __future__ import annotations

import copy
import random

import torch

POLICIES = ("divergent", "convergent", "1sb", "random")


def _mean_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """Mean over nodes of KL(p || q) given log-probabilities."""
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()


class MemoryBank:
    def __init__(self, policy: str, template: torch.nn.Module, temperature: float = 1.0,
                 init_state: dict | None = None, seed: int = 0):
        policy = policy.lower()
        if policy not in POLICIES:
            raise ValueError(f"Unknown memory policy '{policy}'. Choose from {POLICIES}.")
        self.policy = policy
        self.tau = temperature
        self.model = copy.deepcopy(template)  # holds the memory weights
        self.model.eval()
        self.state = None                      # S_m
        self.prev_state = None                 # S_{t-1}
        self.epoch = 0
        self.replacements = 0
        self._rng = random.Random(seed)
        if init_state is not None:             # paper: memory starts as the teacher
            self.state = copy.deepcopy(init_state)
            self.model.load_state_dict(self.state)

    @property
    def ready(self) -> bool:
        return self.state is not None

    @torch.no_grad()
    def log_probs(self, x, edge_index, temperature: float | None = None) -> torch.Tensor:
        """Memory predictions as log-softmax(logits / τ). ``forward`` returns
        log-probabilities, which are logits up to a per-node constant, so dividing
        by τ and re-normalising gives the tempered distribution."""
        tau = self.tau if temperature is None else temperature
        return torch.log_softmax(self.model(x, edge_index) / tau, dim=-1)

    def _load(self, state):
        self.state = copy.deepcopy(state)
        self.model.load_state_dict(self.state)
        self.replacements += 1

    @torch.no_grad()
    def update(self, student: torch.nn.Module, x, edge_index):
        """Call once at the end of every epoch, after the optimizer step."""
        self.epoch += 1
        current = copy.deepcopy(student.state_dict())

        if self.prev_state is not None:
            if self.state is None:
                self._load(self.prev_state)
            elif self.policy == "1sb":
                self._load(self.prev_state)
            elif self.policy == "random":
                if self._rng.random() < 1.0 / self.epoch:  # reservoir sampling
                    self._load(self.prev_state)
            else:
                was_training = student.training
                student.eval()
                log_s = torch.log_softmax(student(x, edge_index) / self.tau, dim=-1)
                student.train(was_training)
                log_m = self.log_probs(x, edge_index)
                prev_model = copy.deepcopy(self.model)
                prev_model.load_state_dict(self.prev_state)
                log_prev = torch.log_softmax(prev_model(x, edge_index) / self.tau, dim=-1)
                kl_mem, kl_prev = _mean_kl(log_s, log_m), _mean_kl(log_s, log_prev)
                if (self.policy == "divergent" and kl_prev > kl_mem) or \
                   (self.policy == "convergent" and kl_prev < kl_mem):
                    self._load(self.prev_state)

        self.prev_state = current
