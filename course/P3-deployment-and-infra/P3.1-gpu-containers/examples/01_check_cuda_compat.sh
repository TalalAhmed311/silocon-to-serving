#!/usr/bin/env bash
# 01_check_cuda_compat.sh — host driver vs container CUDA runtime vs torch build, in one table. T2.
# Usage: bash 01_check_cuda_compat.sh <image>
set -euo pipefail
IMG=${1:?image}
echo "| where | what | value |"; echo "|---|---|---|"
echo "| host | driver (nvidia-smi) | $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) |"
echo "| host | max CUDA the driver supports | $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}') |"
docker run --rm --gpus all --entrypoint python3 "$IMG" -c '
import torch
print("| container | torch.version.cuda |", torch.version.cuda, "|")
print("| container | GPU visible |", torch.cuda.get_device_name() if torch.cuda.is_available() else "NO (driver too old for this runtime?)", "|")
' || echo "| container | python/torch | not available in this image |"
