#!/usr/bin/env bash
# 03_vllm_tp.sh — vLLM with --tensor-parallel-size 1, 2, 4 on a 4-GPU box; batch-1 ITL and the #4 knee for each (T3).
# Run on the instance from aws.md (g6.12xlarge-class, 4× L4). MODEL = a local dir with your pinned 8B weights (P3.7).
# Expected shape: ITL at batch 1 improves sub-linearly with TP (weights read per GPU shrink by tp, but each token now
# pays 2 all-reduces per layer); throughput per GPU usually DROPS vs TP=1 replicas. TODO(run-on: g6.12xlarge)
set -euo pipefail
MODEL=${MODEL:?set MODEL=/path/to/model}
: "${S2S_API_KEY:?export S2S_API_KEY first}"
mkdir -p results/p4.2
for tp in 1 2 4; do
  vllm serve "$MODEL" --tensor-parallel-size "$tp" --host 127.0.0.1 --port 8000 --api-key "$S2S_API_KEY" \
    --max-model-len 4096 --served-model-name m > "results/p4.2/vllm-tp$tp.log" 2>&1 &
  pid=$!
  until curl -sf -H "Authorization: Bearer $S2S_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
  PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 0.2 1 4 8 16 --duration 90 \
    --out "results/p4.2/tp$tp.json"
  kill "$pid"; wait "$pid" || true
done
echo "compare: results/p4.2/tp{1,2,4}.json — ITL at the lowest rate, knee out tok/s, and per-GPU tok/s"
