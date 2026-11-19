#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mr_jiang/qkr
DATASET_ROOT="$ROOT/datasets/IDSpace"
RUN_ROOT="$ROOT/runs/idspace_e3"
LOG_ROOT="$ROOT/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
exec 9>"$RUN_ROOT/.watch.lock"
flock -n 9 || exit 0
while [[ ! -f "$DATASET_ROOT/download.completed" ]]; do
  sleep 60
done
if [[ -f "$RUN_ROOT/train.started" || -f "$RUN_ROOT/train.completed" ]]; then exit 0; fi
while true; do
  read -r FREE0 FREE1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{a=$1} NR==2{print a,$1}')
  if [[ -n "${FREE0:-}" && -n "${FREE1:-}" ]]; then
    TOTAL_FREE=$((FREE0 + FREE1))
    if (( FREE0 >= 20480 && FREE1 >= 20480 && TOTAL_FREE > 40960 )); then
      echo "GPU memory gate passed $(date -Is): gpu0=${FREE0}MiB gpu1=${FREE1}MiB total=${TOTAL_FREE}MiB"
      break
    fi
    echo "GPU memory gate waiting $(date -Is): gpu0=${FREE0}MiB gpu1=${FREE1}MiB total=${TOTAL_FREE}MiB; require each>=20480MiB and total>40960MiB"
  else
    echo "GPU memory gate waiting $(date -Is): unable to read both GPUs"
  fi
  sleep 60
done
nohup setsid bash "$ROOT/reproduction/idspace_e3_launch.sh" >> "$LOG_ROOT/idspace_e3_launch.log" 2>&1 < /dev/null &
echo "$!" > "$LOG_ROOT/idspace_e3_launch.pid"
date -Is > "$RUN_ROOT/watch.triggered"
