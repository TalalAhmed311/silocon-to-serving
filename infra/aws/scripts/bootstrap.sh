#!/usr/bin/env bash
# bootstrap.sh — run ON the GPU instance (over SSM) to install the course env for a phase and pull a pinned model.
# Usage: bash bootstrap.sh p2 [model_repo] [revision]
# Hardware: T2. Idempotent: safe to re-run.
set -euo pipefail
PHASE=${1:-p2}
MODEL=${2:-}            # e.g. an 8B instruct repo you have accepted the license for
REV=${3:-main}          # pin a commit SHA for reproducible benches
cd "$HOME"
echo "== GPU / driver / CUDA" && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv && (nvcc --version || true)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d silicon-to-serving ] || git clone --depth 1 "${S2S_REPO:-https://github.com/talalahmed311/silocon-to-serving}" silicon-to-serving
cd silicon-to-serving
case "$PHASE" in
  p2) uv venv -q .venv-serving && uv pip install -q -p .venv-serving -r env/requirements-serving.txt ;;
  p5) uv sync -q --extra torch ;;
  *)  uv sync -q ;;
esac
export HF_TOKEN=$(aws ssm get-parameter --name /s2s/hf_token --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
if [ -n "$MODEL" ]; then
  echo "== downloading $MODEL@$REV to /opt/models"
  sudo mkdir -p /opt/models && sudo chown "$USER" /opt/models
  .venv-serving/bin/huggingface-cli download "$MODEL" --revision "$REV" --local-dir "/opt/models/$(basename "$MODEL")"
fi
echo "== done. Next: see the module's aws.md"
