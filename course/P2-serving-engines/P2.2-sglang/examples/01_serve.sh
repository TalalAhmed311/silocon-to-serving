#!/usr/bin/env bash
# 01_serve.sh — start SGLang bound to localhost with an API key; log to sglang.log; wait until healthy.
# Usage: bash 01_serve.sh /opt/models/<model> [extra flags]      Hardware: T2.
set -euo pipefail
MODEL=${1:?model path}; shift || true
: "${S2S_API_KEY:?export S2S_API_KEY first}"
python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 --api-key "$S2S_API_KEY" \
  --context-length "${MAX_MODEL_LEN:-8192}" --mem-fraction-static "${MEM_FRACTION:-0.88}" "$@" > sglang.log 2>&1 &
PID=$!
for _ in $(seq 1 600); do
  kill -0 $PID 2>/dev/null || { echo "SGLang exited:"; tail -30 sglang.log; exit 1; }
  curl -sf -H "Authorization: Bearer $S2S_API_KEY" http://127.0.0.1:30000/v1/models >/dev/null && { echo "ready"; exit 0; }
  sleep 1
done
echo "timed out"; exit 1
