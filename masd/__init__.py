"""Memory Augmented Self-Distillation (MASD) for graph neural networks.

Sharma, Chowdhury & Chandra, "Taking lessons from history: Memory Augmented Self
Distillation for graph neural networks", Data Mining and Knowledge Discovery 40:86 (2026).
"""
from .data import load_dataset, sample_subgraph
from .memory import POLICIES, MemoryBank
from .models import MODELS, build_model
from .trainer import train_masd, train_supervised

__all__ = ["load_dataset", "sample_subgraph", "MemoryBank", "POLICIES", "MODELS",
           "build_model", "train_masd", "train_supervised"]
__version__ = "1.0.0"
