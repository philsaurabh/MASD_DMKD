"""Step 2 — distil a student with Memory Augmented Self-Distillation.

The student is trained on a subgraph sampled with LinkNeighborLoader (Sect. 4.2)
and distils from the teacher saved by ``train_teacher.py`` plus a memory of its
own past snapshots. Run one memory policy, or ``--memory all`` to compare the
four policies (paper Figs. 3–4).

Examples:
    python scripts/train_masd.py --dataset citeseer --teacher checkpoints/teacher_citeseer_gat.pt
    python scripts/train_masd.py --dataset cora --teacher checkpoints/teacher_cora_gat.pt --memory all --with-scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from masd import POLICIES, build_model, load_dataset, sample_subgraph, train_masd, train_supervised  # noqa: E402
from masd.evaluate import evaluate  # noqa: E402
from masd.plots import distribution_plot, node_probability_plot, training_curves  # noqa: E402
from masd.utils import get_device, load_checkpoint, save_checkpoint, save_json, set_seed  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="citeseer")
    g.add_argument("--data-root", default="data")
    g.add_argument("--sampling", choices=["linkneighbor", "full"], default="linkneighbor",
                   help="train the student on a sampled subgraph (paper, notebooks) or the full graph")
    g.add_argument("--num-neighbors", type=int, default=100)
    g.add_argument("--num-hops", type=int, default=10)
    g.add_argument("--batch-size", type=int, default=512)

    g = p.add_argument_group("models")
    g.add_argument("--teacher", required=True, help="checkpoint from train_teacher.py")
    g.add_argument("--student-model", default=None, help="default: same architecture as the teacher")
    g.add_argument("--hidden", type=int, default=None)
    g.add_argument("--heads", type=int, default=None)
    g.add_argument("--dropout", type=float, default=None)

    g = p.add_argument_group("MASD")
    g.add_argument("--memory", default="divergent", help=f"{' | '.join(POLICIES)} | all")
    g.add_argument("--uncertainty", default="entropy", help="entropy | mc_dropout | margin | least_confidence")
    g.add_argument("--mc-passes", type=int, default=10)
    g.add_argument("--loss-impl", choices=["paper", "notebook"], default="paper",
                   help="'paper' = Eqs. 2–9; 'notebook' = loss exactly as in the research notebooks")
    g.add_argument("--memory-init", choices=["teacher", "none"], default="teacher")
    g.add_argument("--alpha", type=float, default=1.0)
    g.add_argument("--beta", type=float, default=0.005)
    g.add_argument("--gamma", type=float, default=0.55)
    g.add_argument("--temperature", type=float, default=10.0)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=250)
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--weight-decay", type=float, default=7e-4)
    g.add_argument("--patience", type=int, default=50, help="0 disables early stopping")
    g.add_argument("--warmup", type=int, default=0)
    g.add_argument("--select-on", choices=["val", "test"], default="val",
                   help="checkpoint selection mask; the notebooks used 'test'")

    g = p.add_argument_group("run")
    g.add_argument("--with-scratch", action="store_true", help="also train the student from scratch (CE only)")
    g.add_argument("--sample-node", type=int, default=180, help="node for the probability bar chart")
    g.add_argument("--no-flops", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="auto")
    g.add_argument("--out", default="results")
    return p


def main():
    a = build_parser().parse_args()
    set_seed(a.seed)
    device = get_device(a.device)

    # ---- data: full graph for the teacher, sampled subgraph for the student
    full, in_dim, n_cls = load_dataset(a.dataset, a.data_root, split_seed=a.seed)
    data = full if a.sampling == "full" else sample_subgraph(
        full, a.num_neighbors, a.num_hops, a.batch_size, seed=a.seed)
    data = data.to(device)
    print(f"[data] {a.dataset}: {data.num_nodes} nodes, {data.edge_index.size(1)} edges "
          f"(train {int(data.train_mask.sum())}, val {int(data.val_mask.sum())}, test {int(data.test_mask.sum())})")

    # ---- teacher
    ckpt = load_checkpoint(a.teacher, map_location=device)
    tm = ckpt["meta"]
    t_model = tm.get("model", "gat")
    teacher = build_model(t_model, in_dim, n_cls, hidden=tm.get("hidden"), heads=tm.get("heads"),
                          dropout=tm.get("dropout")).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()

    def new_student():
        return build_model(a.student_model or t_model, in_dim, n_cls,
                           hidden=a.hidden if a.hidden is not None else tm.get("hidden"),
                           heads=a.heads if a.heads is not None else tm.get("heads"),
                           dropout=a.dropout if a.dropout is not None else tm.get("dropout")).to(device)

    out_dir = Path(a.out) / f"{a.dataset}_{a.student_model or t_model}"
    policies = list(POLICIES) if a.memory == "all" else [a.memory]
    results, histories, preds = {}, {}, {}
    test = data.test_mask

    with torch.no_grad():
        preds["Ground truth"] = data.y[test].cpu()
        preds["Teacher"] = teacher(data.x, data.edge_index).argmax(-1)[test].cpu()
    results["teacher"] = evaluate(teacher, data, n_cls, flops=not a.no_flops)

    if a.with_scratch:
        set_seed(a.seed)
        scratch, hist = train_supervised(new_student(), data, epochs=a.epochs, lr=a.lr,
                                         weight_decay=a.weight_decay, patience=a.patience or None,
                                         verbose=False)
        results["scratch"] = evaluate(scratch, data, n_cls, teacher=teacher, flops=not a.no_flops)
        histories["Scratch"] = hist

    for policy in policies:
        set_seed(a.seed)
        student, memory, hist = train_masd(
            new_student(), teacher, data, memory_policy=policy, epochs=a.epochs, lr=a.lr,
            weight_decay=a.weight_decay, alpha=a.alpha, beta=a.beta, gamma=a.gamma,
            temperature=a.temperature, uncertainty=a.uncertainty, mc_passes=a.mc_passes,
            loss_impl=a.loss_impl, memory_init=a.memory_init, patience=a.patience or None,
            select_on=a.select_on, warmup=a.warmup, seed=a.seed)
        results[policy] = evaluate(student, data, n_cls, teacher=teacher, flops=not a.no_flops)
        results[policy]["memory_replacements"] = memory.replacements
        histories[policy] = hist

        save_checkpoint(out_dir / f"student_{policy}.pt", student, {**vars(a), "memory": policy})
        if memory.ready:
            save_checkpoint(out_dir / f"memory_{policy}.pt", memory.model, {"policy": policy})

        with torch.no_grad():
            s_pred = student(data.x, data.edge_index).argmax(-1)[test].cpu()
            series = {"Ground truth": preds["Ground truth"], "Teacher": preds["Teacher"]}
            if memory.ready:
                series["Memory"] = memory.model(data.x, data.edge_index).argmax(-1)[test].cpu()
            series["Student"] = s_pred
            preds[f"Student ({policy})"] = s_pred
            distribution_plot(series, out_dir / f"distribution_{policy}.png", title=f"{policy} memory")
            if a.sample_node < data.num_nodes:
                probs = student(data.x, data.edge_index)[a.sample_node].exp().cpu().numpy()
                node_probability_plot(probs, a.sample_node, out_dir / f"node{a.sample_node}_{policy}.png")

    if len(policies) > 1:
        distribution_plot({k: v for k, v in preds.items()}, out_dir / "distribution_all.png",
                          title="All memory policies")
    training_curves(histories, out_dir / "test_accuracy_per_epoch.png")

    save_json(out_dir / "results.json", {"args": vars(a), "results": results, "histories": histories})
    print("\n" + f"{'run':<12}{'acc':>8}{'F1':>8}{'ECE':>8}{'CKA':>8}")
    for name, r in results.items():
        cka = r.get("cka_vs_teacher")
        print(f"{name:<12}{r['accuracy']:>8.4f}{r['f1_macro']:>8.4f}{r['ece']:>8.4f}"
              f"{(f'{cka:.4f}' if cka is not None else '—'):>8}")
    print(f"\nresults, checkpoints and figures in {out_dir}/")


if __name__ == "__main__":
    main()
