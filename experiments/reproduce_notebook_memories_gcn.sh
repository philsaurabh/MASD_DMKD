#!/usr/bin/env bash
# Reproduce the four memory cells of the GAT notebook (divergent, convergent,
# 1-step back, random) with a GCN student, entropy uncertainty and the
# notebook form of the loss. Hyperparameters follow the convergent / 1-SB /
# random cells: lr 0.1, weight decay 5e-4, alpha 0.5, beta 0.25, gamma 0.25,
# tau 100, 501 epochs. (The divergent cell used alpha 1.0, beta 0.05, gamma 0.5.)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=${DATASET:-citeseer}

python scripts/train_teacher.py --dataset "$DATASET" --model gcn --hidden 64

python scripts/train_masd.py --dataset "$DATASET" --teacher "checkpoints/teacher_${DATASET}_gcn.pt" \
  --memory all --uncertainty entropy --loss-impl notebook --select-on test \
  --alpha 0.5 --beta 0.25 --gamma 0.25 --temperature 100 \
  --lr 0.1 --weight-decay 5e-4 --epochs 501 --patience 0 --with-scratch --out results/notebook
