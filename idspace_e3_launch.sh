#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mr_jiang/qkr
DATASET_ROOT="$ROOT/datasets/IDSpace"
PREPARED_ROOT="$ROOT/runs/idspace_e3/prepared"
RUN_ROOT="$ROOT/runs/idspace_e3_full_metrics"
LOG_ROOT="$ROOT/logs"
PYTHON=/home/mr_jiang/qkr/envs/mineru421/bin/python
PREPARE_PYTHON=/usr/bin/python3
EXTRA=/home/mr_jiang/qkr/envs/idspace_e3_extra
SITE=/home/mr_jiang/qkr/envs/mineru421/lib/python3.12/site-packages
MODEL="$ROOT/models/xiaomi_ocr0/Xiaomi-OCR-0"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
exec 9>"$RUN_ROOT/.e3.lock"
flock -n 9 || exit 0
if [[ -f "$RUN_ROOT/train.completed" || -f "$RUN_ROOT/train.started" ]]; then exit 0; fi

export PYTHONPATH="$EXTRA:$SITE"
echo "E3 launch started $(date -Is)"
"$PREPARE_PYTHON" "$ROOT/reproduction/idspace_e3_prepare.py" --root "$DATASET_ROOT" --output "$PREPARED_ROOT" --seed 42
date -Is > "$RUN_ROOT/train.started"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29543 \
  "$ROOT/reproduction/idspace_e3_train.py" --manifest "$PREPARED_ROOT" --model "$MODEL" --output "$RUN_ROOT" \
  --epochs 1 --lr 1e-4 --batch-size 2 --grad-accum 4 --max-pixels 65536 --seed 42
date -Is > "$RUN_ROOT/train.completed"
echo "E3 launch completed $(date -Is)"


