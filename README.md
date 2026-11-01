# MASD: Memory Augmented Self-Distillation for Graph Neural Networks

Official code for

> **Taking lessons from history: Memory Augmented Self Distillation for graph neural networks**
> Saurabh Sharma, Souvik Chowdhury, Joydeep Chandra
> *Data Mining and Knowledge Discovery* 40, 86 (2026) · [doi:10.1007/s10618-026-01258-z](https://doi.org/10.1007/s10618-026-01258-z)

GNNs overfit easily, and knowledge distillation helps only when a well-trained teacher is available. Over-smoothing makes such teachers hard to obtain. MASD distils a GNN from **its own past**: the student learns from an arbitrarily trained teacher **and** from a *memory*, a stored snapshot of an earlier version of itself. For every node, the student's **prediction uncertainty** decides which source to trust. When the student is uncertain it leans on memory; when it is confident it leans on the teacher. The teacher also acts as a regulariser throughout. Across benchmarks this improves top-1 accuracy by 2.5–6% over standard GNN training and self-distillation baselines.

---

**Training data (Sect. 4.2).** The teacher is trained on the full graph. The student is trained on a subgraph drawn with `LinkNeighborLoader`, using 100 neighbours per hop over 10 hops and a batch of 512 seed edges.

## Repository layout

```
MASD/
├── masd/                      # the library
│   ├── data.py                # Planetoid / Amazon loading, per-class splits, LinkNeighborLoader sampling
│   ├── models.py              # GAT, GAT3, GCN, GraphSAGE, MLP, GraphTransformer (log-prob output + embeddings)
│   ├── uncertainty.py         # entropy, MC dropout, margin, least confidence
│   ├── memory.py              # MemoryBank: divergent / convergent / 1-step back / random policies
│   ├── losses.py              # Eq. 9 objective, "paper" and "notebook" implementations
│   ├── trainer.py             # teacher training and the MASD loop (Algorithm 1)
│   ├── metrics.py             # accuracy, macro P/R/F1, ECE, linear & RBF CKA, fvcore GFLOPs
│   ├── evaluate.py            # one-call evaluation (Table 5 metrics)
│   ├── plots.py               # class-distribution KDE plots (Fig. 3), node probability bars, curves
│   └── utils.py               # seeding, device, checkpoint / JSON I/O
├── scripts/
│   ├── train_teacher.py       # step 1: arbitrarily trained teacher T
│   └── train_masd.py          # step 2: MASD student (one policy or all four)
├── experiments/               # shell recipes for the paper's tables and figures
├── tests/test_smoke.py        # fast tests on a random graph (no downloads)
├── notebooks/                 # original research notebooks (provenance)
├── requirements.txt · pyproject.toml · CITATION.cff
```

## Installation

Python ≥ 3.9. PyTorch ≥ 2.1 and PyTorch Geometric ≥ 2.5 are required; the notebooks ran on PyTorch 2.10 (CUDA 12.8).

```bash
git clone https://github.com/<your-username>/MASD.git
cd MASD
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch for your CUDA version: see https://pytorch.org/get-started/locally/
pip install torch

# 2. PyTorch Geometric and a neighbour-sampling backend (needed by LinkNeighborLoader)
pip install torch_geometric
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', ''))")
pip install pyg_lib torch_sparse -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"

# 3. Everything else
pip install -r requirements.txt
pip install -e .            # optional: makes `import masd` work from anywhere
```

Check the setup without downloading any data:

```bash
pytest -q tests/
```

Datasets download automatically into `data/` the first time they are used.

## Quick start

```bash
# 1. Train the arbitrarily trained teacher (2-layer GAT, 64 hidden × 8 heads) on Citeseer
python scripts/train_teacher.py --dataset citeseer --model gat

# 2. Distil a student with divergent memory
python scripts/train_masd.py --dataset citeseer --teacher checkpoints/teacher_citeseer_gat.pt --memory divergent

# 3. Compare all four memory policies against a student trained from scratch
python scripts/train_masd.py --dataset citeseer --teacher checkpoints/teacher_citeseer_gat.pt \
    --memory all --with-scratch
```

Each run prints a summary table of accuracy, macro F1, ECE and CKA for the teacher, the scratch student (if requested) and every memory policy. Checkpoints, metrics and figures are written to `results/<dataset>_<model>/` (see [Outputs](#outputs)).

## Reproducing the experiments

Each script in `experiments/` is a self-contained recipe:

| Script | Reproduces | What it runs |
|---|---|---|
| `node_classification_table3.sh` | Table 3 | GCN and GAT with divergent memory on Cora, Citeseer, Pubmed, Amazon Photo, Amazon Computers; 5 seeds |
| `memory_comparison_fig4.sh` | Figs. 3–4 | all four memory policies × GCN, GAT, GraphSAGE |
| `backbones_table5.sh` | Table 5 | GraphSAGE, Graph MLP and Graph Transformer on Amazon Computers |
| `uncertainty_fig6.sh` | Fig. 6 | entropy vs MC dropout vs margin vs least confidence |
| `reproduce_notebook_citeseer_gat.sh` | notebook GAT cells | GAT student, MC dropout, notebook loss, test-set selection |
| `reproduce_notebook_memories_gcn.sh` | notebook memory cells | GCN student, all four memories, notebook hyperparameters |

```bash
bash experiments/node_classification_table3.sh
SEEDS="0 1 2" bash experiments/node_classification_table3.sh   # fewer seeds
DATASET=cora bash experiments/reproduce_notebook_memories_gcn.sh
```

**Hyperparameters.** Section 5.1 of the paper tunes Adam over learning rate ∈ {0.001, 0.0015, 0.005, 0.01}, weight decay ∈ {0.0005, 0.005, 0.001, 0.01} and temperature τ ∈ {1, 10, 50}. It trains for 250 epochs with early stopping (patience 50) on validation accuracy. The script defaults (lr 0.01, weight decay 7e-4, α = 1.0, β = 0.005, γ = 0.55, τ = 10) are the notebook's GAT settings. Best values differ by dataset and backbone, so sweep the grid above to match the reported numbers.

**Splits.** Cora, Citeseer and Pubmed use the public Planetoid splits. Amazon Photo and Computers have no official split; this code uses 20 training and 30 validation nodes per class, with the rest as test (`masd.data.random_class_split`, seeded by `--seed`).

## Command-line reference

`scripts/train_teacher.py`

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `citeseer` | `cora`, `citeseer`, `pubmed`, `photo`, `computers` |
| `--model` | `gat` | `gat`, `gat3`, `gcn`, `sage`, `mlp`, `transformer` |
| `--hidden` / `--heads` / `--dropout` | 64 / 8 / 0.5 | architecture (heads only for attention models) |
| `--epochs` / `--lr` / `--weight-decay` | 501 / 0.01 / 1e-3 | optimisation, as in the notebooks |
| `--patience` | off | optional early stopping on validation accuracy |

`scripts/train_masd.py`

| Flag | Default | Meaning |
|---|---|---|
| `--teacher` | required | checkpoint from `train_teacher.py` |
| `--memory` | `divergent` | `divergent`, `convergent`, `1sb`, `random`, or `all` |
| `--uncertainty` | `entropy` | `entropy`, `mc_dropout`, `margin`, `least_confidence` |
| `--loss-impl` | `paper` | `paper` (Eqs. 2–9) or `notebook` (loss exactly as in the notebooks) |
| `--alpha` / `--beta` / `--gamma` | 1.0 / 0.005 / 0.55 | loss weights in Eq. 9 |
| `--temperature` | 10 | distillation temperature τ |
| `--epochs` / `--patience` | 250 / 50 | `--patience 0` disables early stopping |
| `--lr` / `--weight-decay` | 0.01 / 7e-4 | Adam |
| `--warmup` | 0 | epochs of cross-entropy-only training before distillation |
| `--select-on` | `val` | checkpoint selection mask (`test` mirrors the notebooks) |
| `--sampling` | `linkneighbor` | train the student on a sampled subgraph or the `full` graph |
| `--memory-init` | `teacher` | start memory from the teacher's weights, or `none` |
| `--student-model` | teacher's | a different student backbone |
| `--with-scratch` | off | also train a cross-entropy-only student for comparison |
| `--no-flops` | off | skip the fvcore GFLOPs count |

## Outputs

```
checkpoints/
  teacher_<dataset>_<model>.pt        # teacher weights + metadata
  teacher_<dataset>_<model>.json      # teacher metrics and training history
results/<dataset>_<model>/
  results.json                        # all metrics and per-epoch histories
  student_<policy>.pt                 # best student
  memory_<policy>.pt                  # final memory snapshot
  distribution_<policy>.png           # ground truth vs teacher vs memory vs student (Fig. 3a–d)
  distribution_all.png                # all policies together (Fig. 3e), with --memory all
  test_accuracy_per_epoch.png         # per-epoch test accuracy (Fig. 3f)
  node<k>_<policy>.png                # class probabilities for node k
```

Metrics reported: top-1 accuracy, macro precision, recall and F1, expected calibration error (15 bins), linear CKA between student and teacher embeddings, inference time and GFLOPs. fvcore does not count PyG's scatter and gather operators, so treat its GFLOPs as a lower bound.


## Scope

This release covers **node classification** (Tables 3 and 5, Figs. 3, 4 and 6). The notebooks do not include the following, so they are not here:

- graph classification on DD and PROTEINS (Table 4);
- the PPI teacher–student comparison (Fig. 5a);
- the Elliptic Bitcoin experiments;
- the depth study (Fig. 10).

The `MemoryBank` and `masd_loss` components are model- and task-agnostic. To extend to graph classification, apply them to per-graph logits from a pooled GNN.

## Citation

```bibtex
@article{sharma2026masd,
  title   = {Taking lessons from history: Memory Augmented Self Distillation for graph neural networks},
  author  = {Sharma, Saurabh and Chowdhury, Souvik and Chandra, Joydeep},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {40},
  pages   = {86},
  year    = {2026},
  doi     = {10.1007/s10618-026-01258-z}
}
```

## Acknowledgements

The CKA implementation is adapted from [yuanli2333/CKA-Centered-Kernel-Alignment](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment). Models and data loaders are built on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).

## Contact

please contact saurabh_2021cs30@iitp.ac.in for any discrepancy.


