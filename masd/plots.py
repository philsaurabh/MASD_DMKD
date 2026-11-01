"""Figures from the notebooks: class-distribution KDE plots (paper Fig. 3) and the
predicted-probability bar chart for a single node."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

PALETTE = ["maroon", "indigo", "darkgreen", "darkorange", "royalblue", "goldenrod"]


def distribution_plot(series: dict, path: str | Path, title: str | None = None) -> None:
    """KDE of predicted class labels on test nodes, one curve per source
    (e.g. ground truth, teacher, memory, student)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    for (label, values), color in zip(series.items(), PALETTE):
        sns.kdeplot(np.asarray(values), color=color, label=label, linewidth=3)
    plt.legend(fontsize=18, loc="upper right")
    plt.xlabel("Node classes", fontsize=25)
    plt.ylabel("Proportion of classes", fontsize=25)
    if title:
        plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def node_probability_plot(probs, node: int, path: str | Path) -> None:
    """Bar chart of predicted class probabilities for one node."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.barplot(x=np.arange(len(probs)), y=np.asarray(probs), color="steelblue")
    plt.xlabel("Class")
    plt.ylabel("Prediction probability")
    plt.title(f"Predicted class probabilities for node {node}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def training_curves(histories: dict, path: str | Path, key: str = "test_acc") -> None:
    """Per-epoch accuracy for several runs (paper Fig. 3f)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, hist in histories.items():
        plt.plot([h["epoch"] for h in hist], [h[key] for h in hist], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(key.replace("_", " ").title())
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
