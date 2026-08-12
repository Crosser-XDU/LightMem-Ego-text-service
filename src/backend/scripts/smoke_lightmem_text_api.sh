#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8767}"
USER_ID="${USER_ID:-smoke-user}"
SESSION_ID="${SESSION_ID:-smoke-session}"
STAMP="$(date +%s%3N)"

curl -sS --max-time 10 "${BASE_URL}/health" >/dev/null
curl -sS --max-time 20 -X POST "${BASE_URL}/add" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"smoke-${STAMP}\",\"user_id\":\"${USER_ID}\",\"session_id\":\"${SESSION_ID}\",\"messages\":[{\"role\":\"user\",\"timestamp\":${STAMP},\"content\":\"The smoke test memory says the deployment keyword is amber-light.\"}]}" >/dev/null
curl -sS --max-time 60 -X POST "${BASE_URL}/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"What is the deployment keyword?\",\"user_id\":\"${USER_ID}\",\"session_id\":\"${SESSION_ID}\",\"top_k\":3}"
printf "\n"
