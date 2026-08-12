#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/text_embedding runtime/text_embedding
PID_PATH="runtime/text_embedding/server.pid"
LOG_PATH="logs/text_embedding/server.log"

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    cmdline="$(tr '\0' ' ' < "/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"text_embedding_server.py"* ]]; then
      echo "[start_text_embedding_server] server already running pid=${existing_pid}; reusing it"
      exit 0
    fi
    echo "[start_text_embedding_server] pid file points to another process: ${existing_pid}" >&2
    exit 1
  fi
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export CUDA_VISIBLE_DEVICES="${LIGHTMEM_TEXT_EMBED_CUDA_VISIBLE_DEVICES:-${LIGHTMEM_TEXT_CUDA_VISIBLE_DEVICES:-4}}"
export LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST="${LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST:-127.0.0.1}"
export LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT="${LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT:-8010}"
export LIGHTMEM_TEXT_EMBEDDING_MODEL_PATH="${LIGHTMEM_TEXT_EMBEDDING_MODEL_PATH:-models/all-MiniLM-L6-v2}"
export LIGHTMEM_TEXT_EMBEDDING_DEVICE="${LIGHTMEM_TEXT_EMBEDDING_DEVICE:-cuda}"
export LIGHTMEM_TEXT_EMBEDDING_BATCH_SIZE="${LIGHTMEM_TEXT_EMBEDDING_BATCH_SIZE:-128}"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

setsid "$python_bin" text_embedding_server.py \
  --model "$LIGHTMEM_TEXT_EMBEDDING_MODEL_PATH" \
  --device "$LIGHTMEM_TEXT_EMBEDDING_DEVICE" \
  --batch-size "$LIGHTMEM_TEXT_EMBEDDING_BATCH_SIZE" \
  --host "$LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST" \
  --port "$LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT" \
  > "$LOG_PATH" 2>&1 < /dev/null &
echo "$!" > "$PID_PATH"

echo "[start_text_embedding_server] pid=$(cat "$PID_PATH") log=${LOG_PATH}"
echo "[start_text_embedding_server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[start_text_embedding_server] url=http://${LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST}:${LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT}/v1"

for _ in {1..60}; do
  if curl -sS --max-time 2 "http://${LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST}:${LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT}/health" >/dev/null 2>&1; then
    curl -sS --max-time 5 "http://${LIGHTMEM_TEXT_EMBEDDING_SERVER_HOST}:${LIGHTMEM_TEXT_EMBEDDING_SERVER_PORT}/health"
    printf "\n"
    exit 0
  fi
  sleep 1
done

echo "[start_text_embedding_server] failed to become healthy; tailing log" >&2
tail -80 "$LOG_PATH" >&2 || true
exit 1
