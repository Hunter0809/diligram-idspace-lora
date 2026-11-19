#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mr_jiang/qkr"
LOG_ROOT="$ROOT/logs"
PID_FILE="$LOG_ROOT/idspace_download.pid"
mkdir -p "$LOG_ROOT" "$ROOT/reproduction"

if [[ "${IDSPACE_DOWNLOAD_DAEMON:-0}" != "1" ]]; then
    PERSIST="$ROOT/reproduction/idspace_download.sh"
    if [[ "$(readlink -f "$0")" != "$(readlink -f "$PERSIST")" ]]; then
        cp "$0" "$PERSIST"
        chmod 755 "$PERSIST"
    fi
    if [[ -s "$PID_FILE" ]]; then
        existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
            echo "IDSpace download already running: pid=$existing_pid"
            exit 0
        fi
    fi
    nohup setsid env IDSPACE_DOWNLOAD_DAEMON=1 bash "$PERSIST" \
        >> "$LOG_ROOT/idspace_download.log" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "IDSpace download started: pid=$! log=$LOG_ROOT/idspace_download.log"
    exit 0
fi

exec /usr/bin/python3 "$ROOT/reproduction/idspace_parallel_download.py"
