#!/usr/bin/env bash
# Reproduce the GAT experiment cells of the notebooks (Citeseer):
# GAT teacher (hidden 64, 8 heads) -> GAT student with divergent memory,
# Monte Carlo dropout uncertainty, the notebook form of the loss, and
# checkpoint selection on test accuracy, exactly as the notebooks did.
# The notebooks trained the student for only 2 epochs; 250 epochs (Sect. 5.1) are used here.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_teacher.py --dataset citeseer --model gat --hidden 64 --heads 8 --dropout 0.5 \
  --epochs 501 --lr 0.01 --weight-decay 1e-3

python scripts/train_masd.py --dataset citeseer --teacher checkpoints/teacher_citeseer_gat.pt \
  --memory divergent --uncertainty mc_dropout --loss-impl notebook --select-on test \
  --alpha 1.0 --beta 0.005 --gamma 0.55 --temperature 10 \
  --lr 0.01 --weight-decay 7e-4 --epochs 250 --patience 0 --out results/notebook
