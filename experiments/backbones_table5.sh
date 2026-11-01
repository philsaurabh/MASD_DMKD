#!/usr/bin/env bash
# Different backbones on Amazon Computers with divergent memory (paper Table 5):
# accuracy, macro F1, ECE, CKA vs teacher, inference time and GFLOPs.
set -euo pipefail
cd "$(dirname "$0")/.."

for model in sage mlp transformer; do
  extra=""
  [ "$model" = "mlp" ] || [ "$model" = "transformer" ] && extra="--hidden 512"
  python scripts/train_teacher.py --dataset computers --model "$model" $extra
  python scripts/train_masd.py --dataset computers --teacher "checkpoints/teacher_computers_${model}.pt" \
    --memory divergent --out results/table5
done
