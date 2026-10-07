#!/usr/bin/env bash
# #0 v1 vs vLLM with #4 loadgen on the same GPU and model (P6.7). T2. TODO(run-on: g6.xlarge)
# Usage: MODEL_DIR=build/my-model HF_MODEL=<hf id> bash course/P6-engine-internals/P6.7-project-tiny-engine-gpu/bench/compare.sh
set -euo pipefail
: "${MODEL_DIR:?converted model dir for #0 v1}" "${HF_MODEL:?same model's HF id or path for vLLM}"
: "${S2S_API_KEY:=$(openssl rand -hex 16)}"; export S2S_API_KEY
MAX_SEQS=${MAX_SEQS:-64}; MAX_TOKENS=${MAX_TOKENS:-2048}; MAX_LEN=${MAX_LEN:-4096}
RATES=${RATES:-"0.5 1 2 4 8 16"}; DURATION=${DURATION:-120}
mkdir -p results/p6.7

run_load () {   # $1 = url, $2 = out
  PYTHONPATH=platform uv run python -m loadgen.cli --url "$1" --rates $RATES --duration "$DURATION" \
    --prompt-mean 512 --output-mean 128 --out "$2"
}

echo "== vLLM (max_num_seqs=$MAX_SEQS, max_num_batched_tokens=$MAX_TOKENS, max_model_len=$MAX_LEN, bf16)"
vllm serve "$HF_MODEL" --host 127.0.0.1 --port 8000 --api-key "$S2S_API_KEY" --dtype bfloat16 \
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_TOKENS" --max-model-len "$MAX_LEN" &
VLLM=$!; trap 'kill $VLLM 2>/dev/null || true' EXIT
until curl -sf -H "Authorization: Bearer $S2S_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
run_load http://127.0.0.1:8000 results/p6.7/vllm.json
kill $VLLM; wait $VLLM 2>/dev/null || true

echo "== #0 v1 (same limits)"
S2S_RUNNER=torch S2S_MODEL="$MODEL_DIR" S2S_MAX_NUM_SEQS=$MAX_SEQS S2S_MAX_BATCHED_TOKENS=$MAX_TOKENS \
  uv run --extra torch uvicorn s2s_engine.server:app --app-dir platform/engine/v1 --host 127.0.0.1 --port 8002 &
S2S=$!; trap 'kill $S2S 2>/dev/null || true' EXIT
until curl -sf http://127.0.0.1:8002/health >/dev/null; do sleep 5; done
run_load http://127.0.0.1:8002 results/p6.7/s2s.json
kill $S2S

PYTHONPATH=platform uv run python -m loadgen.plot results/p6.7/vllm.json results/p6.7/s2s.json
