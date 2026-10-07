#!/usr/bin/env bash
# 01_serve.sh — start vLLM bound to localhost with an API key, log to vllm.log, wait until healthy.
# Usage: bash 01_serve.sh /opt/models/<model> [extra vllm flags...]
# Hardware: T2. Stops with a clear message if the server dies during start-up (OOM, bad flag, ...).
set -euo pipefail
MODEL=${1:?model path}; shift || true
: "${S2S_API_KEY:?export S2S_API_KEY first (e.g. openssl rand -hex 16)}"
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --api-key "$S2S_API_KEY" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" "$@" > vllm.log 2>&1 &
PID=$!
echo "vllm pid $PID, logging to vllm.log"
for _ in $(seq 1 600); do
  if ! kill -0 $PID 2>/dev/null; then echo "vLLM exited during start-up; last lines:"; tail -30 vllm.log; exit 1; fi
  if curl -sf -H "Authorization: Bearer $S2S_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "ready after ~$SECONDS s"; grep -iE "kv cache|concurrency|graph" vllm.log | tail -5 || true; exit 0
  fi
  sleep 1
done
echo "timed out waiting for vLLM"; exit 1
