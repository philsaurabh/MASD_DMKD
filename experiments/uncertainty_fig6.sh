#!/usr/bin/env bash
# Effect of the uncertainty measure (paper Sect. 6.4, Fig. 6): two-layer GAT on
# Amazon Computers with divergent memory.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_teacher.py --dataset computers --model gat
for u in entropy mc_dropout margin least_confidence; do
  python scripts/train_masd.py --dataset computers --teacher checkpoints/teacher_computers_gat.pt \
    --memory divergent --uncertainty "$u" --out "results/fig6_$u"
done
