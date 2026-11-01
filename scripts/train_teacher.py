"""Step 1 — train the arbitrarily trained teacher T on the full graph.

Example:
    python scripts/train_teacher.py --dataset citeseer --model gat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masd import build_model, load_dataset, train_supervised  # noqa: E402
from masd.evaluate import evaluate  # noqa: E402
from masd.utils import get_device, save_checkpoint, save_json, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="citeseer", help="cora | citeseer | pubmed | photo | computers")
    p.add_argument("--model", default="gat", help="gat | gat3 | gcn | sage | mlp | transformer")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=501)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping on validation accuracy (default: off, as in the notebooks)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints", help="output directory")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    set_seed(a.seed)
    device = get_device(a.device)
    data, in_dim, n_cls = load_dataset(a.dataset, a.data_root, split_seed=a.seed)
    data = data.to(device)
    model = build_model(a.model, in_dim, n_cls, hidden=a.hidden, heads=a.heads, dropout=a.dropout).to(device)

    model, history = train_supervised(model, data, epochs=a.epochs, lr=a.lr,
                                      weight_decay=a.weight_decay, patience=a.patience)
    metrics = evaluate(model, data, n_cls)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    tag = f"{a.dataset}_{a.model}"
    meta = {"dataset": a.dataset, "model": a.model, "hidden": a.hidden, "heads": a.heads,
            "dropout": a.dropout, "seed": a.seed}
    save_checkpoint(Path(a.out) / f"teacher_{tag}.pt", model, meta)
    save_json(Path(a.out) / f"teacher_{tag}.json", {"args": vars(a), "metrics": metrics, "history": history})
    print(f"saved {Path(a.out) / f'teacher_{tag}.pt'}")


if __name__ == "__main__":
    main()
