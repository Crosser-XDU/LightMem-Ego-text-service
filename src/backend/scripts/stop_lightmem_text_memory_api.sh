#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PID_PATH="runtime/text_memory_api/server.pid"
if [[ ! -f "$PID_PATH" ]]; then
  echo "[stop_lightmem_text_memory_api] no pid file"
  exit 0
fi
pid="$(cat "$PID_PATH" 2>/dev/null || true)"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "[stop_lightmem_text_memory_api] not running"
  rm -f "$PID_PATH"
  exit 0
fi
cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
if [[ "$cmdline" != *"lightmem_text_memory_api_server.py"* ]]; then
  echo "[stop_lightmem_text_memory_api] refusing to kill unrelated pid=${pid}" >&2
  exit 1
fi
kill "$pid" 2>/dev/null || true
for _ in {1..30}; do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.2
done
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "-$pid" 2>/dev/null || true
fi
rm -f "$PID_PATH"
echo "[stop_lightmem_text_memory_api] stopped pid=${pid}"
