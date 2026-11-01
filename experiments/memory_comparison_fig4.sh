#!/usr/bin/env bash
# Compare the four memory policies across GCN, GAT and GraphSAGE (paper Figs. 3–4).
set -euo pipefail
cd "$(dirname "$0")/.."

for ds in cora citeseer pubmed photo computers; do
  for model in gcn gat sage; do
    python scripts/train_teacher.py --dataset "$ds" --model "$model"
    python scripts/train_masd.py --dataset "$ds" --teacher "checkpoints/teacher_${ds}_${model}.pt" \
      --memory all --with-scratch --out results/fig4
  done
done
