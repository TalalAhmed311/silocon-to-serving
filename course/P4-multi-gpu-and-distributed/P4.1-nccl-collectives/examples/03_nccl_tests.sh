#!/usr/bin/env bash
# 03_nccl_tests.sh — build nccl-tests v2.21.1 against the instance's CUDA + NCCL and sweep every collective (T3).
# Run on the multi-GPU instance from aws.md. Output: results/nccl/<collective>.txt, then parse with bench/parse_nccl_tests.py.
# Expected: busbw rises with size and plateaus; the plateau vs the link peak is the number this module is about.
# TODO(run-on: g6.12xlarge, p4d.24xlarge)
set -euo pipefail
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p results/nccl
if [ ! -x nccl-tests/build/all_reduce_perf ]; then
  git clone --depth 1 --branch v2.21.1 https://github.com/NVIDIA/nccl-tests.git    # pinned tag (SOURCES.md)
  make -C nccl-tests -j CUDA_HOME="$CUDA_HOME" ${NCCL_HOME:+NCCL_HOME=$NCCL_HOME}
fi
nvidia-smi topo -m | tee results/nccl/topo.txt                                      # PCIe vs NVLink paths between GPUs
for c in all_reduce reduce_scatter all_gather broadcast alltoall; do
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH ./nccl-tests/build/${c}_perf -b 8 -e 4G -f 2 -g "$NGPU" -w 5 -n 20 \
    2>&1 | tee "results/nccl/${c}.txt"
done
echo "parse: python course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/bench/parse_nccl_tests.py results/nccl/all_reduce.txt --link-gbs <peak>"
