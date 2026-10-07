#!/usr/bin/env bash
# contention.sh — #11 contention matrix on one GPU (T3): N vLLM servers sharing a GPU under a given mode, all loaded at
# once by #4; prints per-tenant tok/s, p99 ITL and Jain's index.
#   MODE=timeslice|mps|mig N=2|4 MODEL=/path bash platform/partitioning/contention.sh
# Time-slicing on one box = just start N servers on the same GPU (CUDA time-slices them). MPS = start the MPS daemon
# first (nvidia-cuda-mps-control -d). MIG = one server per MIG device (CUDA_VISIBLE_DEVICES=MIG-<uuid>).
# Each server gets --gpu-memory-utilization 0.9/N (time-slice, MPS) so they fit together. TODO(run-on: p4d.24xlarge)
set -euo pipefail
MODE=${MODE:?}; N=${N:-2}; MODEL=${MODEL:?}; : "${S2S_API_KEY:?}"
mkdir -p results/p4.4
[ "$MODE" = mps ] && nvidia-cuda-mps-control -d
mapfile -t MIGS < <(nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*' || true)
pids=()
for i in $(seq 0 $((N - 1))); do
  dev=0; [ "$MODE" = mig ] && dev=${MIGS[$i]}
  util=$(python3 -c "print(0.9 if '$MODE' == 'mig' else round(0.9 / $N, 3))")
  CUDA_VISIBLE_DEVICES=$dev vllm serve "$MODEL" --port $((8100 + i)) --host 127.0.0.1 --api-key "$S2S_API_KEY" \
    --gpu-memory-utilization "$util" --max-model-len 2048 > "results/p4.4/$MODE-$N-server$i.log" 2>&1 &
  pids+=($!)
done
for i in $(seq 0 $((N - 1))); do
  until curl -sf -H "Authorization: Bearer $S2S_API_KEY" http://127.0.0.1:$((8100 + i))/v1/models >/dev/null; do sleep 5; done
done
lpids=()
for i in $(seq 0 $((N - 1))); do
  PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:$((8100 + i)) --rates 4 --duration 120 \
    --out "results/p4.4/$MODE-$N-t$i.json" &
  lpids+=($!)
done
wait "${lpids[@]}"                                  # all tenants loaded at the same time
kill "${pids[@]}" 2>/dev/null || true
[ "$MODE" = mps ] && echo quit | nvidia-cuda-mps-control
PYTHONPATH=platform python -m partitioning.cli table --mode "$MODE" results/p4.4/$MODE-$N-t*.json
