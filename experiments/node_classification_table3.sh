#!/usr/bin/env bash
# Node classification with divergent memory (paper Table 3): GCN and GAT
# backbones on five of the six benchmarks (Elliptic Bitcoin is not included, see README).
# Uses the paper formulation of the loss, entropy uncertainty, validation-based
# early stopping (patience 50) and 250 epochs. Repeat with SEEDS for mean ± std.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4"}

for ds in cora citeseer pubmed photo computers; do
  for model in gcn gat; do
    for seed in $SEEDS; do
      python scripts/train_teacher.py --dataset "$ds" --model "$model" --seed "$seed" \
        --out "checkpoints/seed$seed"
      python scripts/train_masd.py --dataset "$ds" --teacher "checkpoints/seed$seed/teacher_${ds}_${model}.pt" \
        --memory divergent --seed "$seed" --with-scratch --out "results/table3/seed$seed"
    done
  done
done
