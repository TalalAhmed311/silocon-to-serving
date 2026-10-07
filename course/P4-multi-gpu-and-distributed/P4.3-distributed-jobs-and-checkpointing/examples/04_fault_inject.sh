#!/usr/bin/env bash
# 04_fault_inject.sh — kill training mid-run, restart, measure RPO/RTO.
#   local  (T0, CPU): kill rank 1 at step 37 via --die-at-step, then relaunch; prints the RPO/RTO table.
#   ray    (T3): delete one Ray worker pod of the running RayJob; watch Ray restart it and train.py resume.
set -euo pipefail
mode=${1:-local}
mkdir -p results/p4.3
if [ "$mode" = local ]; then
  CK=$(mktemp -d)
  torchrun --nproc-per-node 2 platform/training/train.py --steps 80 --ckpt-every 20 --ckpt "$CK" \
    --die-at-step 37 --die-rank 1 | tee results/p4.3/run1.log || true      # the job dies (exit 17 on rank 1)
  t0=$(date +%s.%N)
  torchrun --nproc-per-node 2 platform/training/train.py --steps 80 --ckpt-every 20 --ckpt "$CK" | tee results/p4.3/run2.log
  echo "relaunch overhead (process start) is inside the RTO's first-step time; restart delay here = 0"
  PYTHONPATH=platform python -m training.rto results/p4.3/run1.log results/p4.3/run2.log --restart-s 0
elif [ "$mode" = ray ]; then
  pod=$(kubectl -n s2s-train get pods -l ray.io/node-type=worker -o name | head -n1)   # label UNVERIFIED for your KubeRay
  date -u +%FT%TZ | tee results/p4.3/ray-kill-time.txt
  kubectl -n s2s-train delete "$pod" --wait=false
  kubectl -n s2s-train logs -f -l ray.io/node-type=head --tail=50 | tee results/p4.3/ray-after-kill.log
else
  echo "usage: $0 [local|ray]"; exit 2
fi
