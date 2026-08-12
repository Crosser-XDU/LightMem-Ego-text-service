#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/text_memory_api runtime/text_memory_api
PID_PATH="runtime/text_memory_api/server.pid"
LOG_PATH="logs/text_memory_api/server.log"

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    cmdline="$(tr '\0' ' ' < "/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"lightmem_text_memory_api_server.py"* ]]; then
      echo "[start_lightmem_text_memory_api] server already running pid=${existing_pid}; reusing it"
      exit 0
    fi
    echo "[start_lightmem_text_memory_api] pid file points to another process: ${existing_pid}" >&2
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

export CUDA_VISIBLE_DEVICES="${LIGHTMEM_TEXT_CUDA_VISIBLE_DEVICES:-4}"
export LIGHTMEM_TEXT_MEMORY_API_HOST="${LIGHTMEM_TEXT_MEMORY_API_HOST:-0.0.0.0}"
export LIGHTMEM_TEXT_MEMORY_API_PORT="${LIGHTMEM_TEXT_MEMORY_API_PORT:-8767}"
export LIGHTMEM_TEXT_MEMORY_API_DATA_DIR="${LIGHTMEM_TEXT_MEMORY_API_DATA_DIR:-runtime/text_memory_api}"
export LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND="${LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND:-auto}"
export LIGHTMEM_TEXT_EMBEDDING_BASE_URL="${LIGHTMEM_TEXT_EMBEDDING_BASE_URL:-http://127.0.0.1:8010/v1}"
export LIGHTMEM_TEXT_EMBEDDING_API_KEY="${LIGHTMEM_TEXT_EMBEDDING_API_KEY:-EMPTY}"
export LIGHTMEM_TEXT_EMBEDDING_MODEL="${LIGHTMEM_TEXT_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
export LIGHTMEM_TEXT_METHOD_MODEL="${LIGHTMEM_TEXT_METHOD_MODEL:-lightmem_text_rule}"
export EM2MEM_MEMORY_GENERATION_BACKEND="${EM2MEM_MEMORY_GENERATION_BACKEND:-rule}"
export EM2MEM_PIPELINE_MODE="${EM2MEM_PIPELINE_MODE:-mst}"
export EM2MEM_MST_RECENT_WINDOW_SECONDS="${EM2MEM_MST_RECENT_WINDOW_SECONDS:-315360000}"
export EM2MEM_MST_MAX_EVENTS="${EM2MEM_MST_MAX_EVENTS:-1000000}"
export EM2MEM_MST_ARCHIVE_MAX_EVENTS="${EM2MEM_MST_ARCHIVE_MAX_EVENTS:-1000000}"
export LIGHTMEM_TEXT_BUILD_ON_ADD="${LIGHTMEM_TEXT_BUILD_ON_ADD:-0}"
export LIGHTMEM_TEXT_SEARCH_CONCURRENCY="${LIGHTMEM_TEXT_SEARCH_CONCURRENCY:-2}"
export LIGHTMEM_TEXT_SEARCH_QUEUE_TIMEOUT="${LIGHTMEM_TEXT_SEARCH_QUEUE_TIMEOUT:-10}"
export LIGHTMEM_TEXT_LT_CACHE_USERS="${LIGHTMEM_TEXT_LT_CACHE_USERS:-8}"
export LIGHTMEM_TEXT_LT_PREFILTER_BASE="${LIGHTMEM_TEXT_LT_PREFILTER_BASE:-96}"
export LIGHTMEM_TEXT_LT_PREFILTER_CAP="${LIGHTMEM_TEXT_LT_PREFILTER_CAP:-256}"
export LIGHTMEM_TEXT_HEALTH_SCAN_LIMIT="${LIGHTMEM_TEXT_HEALTH_SCAN_LIMIT:-64}"
export LIGHTMEM_TEXT_LT_EMBED_BATCH="${LIGHTMEM_TEXT_LT_EMBED_BATCH:-128}"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

setsid "$python_bin" lightmem_text_memory_api_server.py \
  --host "$LIGHTMEM_TEXT_MEMORY_API_HOST" \
  --port "$LIGHTMEM_TEXT_MEMORY_API_PORT" \
  --data-dir "$LIGHTMEM_TEXT_MEMORY_API_DATA_DIR" \
  --embedding-backend "$LIGHTMEM_TEXT_MEMORY_API_EMBEDDING_BACKEND" \
  --embedding-base-url "$LIGHTMEM_TEXT_EMBEDDING_BASE_URL" \
  --embedding-api-key "$LIGHTMEM_TEXT_EMBEDDING_API_KEY" \
  --embedding-model "$LIGHTMEM_TEXT_EMBEDDING_MODEL" \
  --method-model-name "$LIGHTMEM_TEXT_METHOD_MODEL" \
  > "$LOG_PATH" 2>&1 < /dev/null &
echo "$!" > "$PID_PATH"

echo "[start_lightmem_text_memory_api] pid=$(cat "$PID_PATH") log=${LOG_PATH}"
echo "[start_lightmem_text_memory_api] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[start_lightmem_text_memory_api] url=http://127.0.0.1:${LIGHTMEM_TEXT_MEMORY_API_PORT}"

for _ in {1..30}; do
  if curl -sS --max-time 2 "http://127.0.0.1:${LIGHTMEM_TEXT_MEMORY_API_PORT}/health" >/dev/null 2>&1; then
    curl -sS --max-time 5 "http://127.0.0.1:${LIGHTMEM_TEXT_MEMORY_API_PORT}/health"
    printf "\n"
    exit 0
  fi
  sleep 1
done

echo "[start_lightmem_text_memory_api] failed to become healthy; tailing log" >&2
tail -80 "$LOG_PATH" >&2 || true
exit 1
