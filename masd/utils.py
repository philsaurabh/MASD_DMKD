"""Seeding, device selection, checkpoint and result I/O."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(path: str | os.PathLike, model: torch.nn.Module, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if "state_dict" not in ckpt:  # plain state_dict (e.g. from the notebooks)
        ckpt = {"state_dict": ckpt, "meta": {}}
    return ckpt


def save_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
