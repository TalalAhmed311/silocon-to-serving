#!/usr/bin/env bash
# ci_terraform.sh — offline checks for every Terraform root under infra/: fmt, init (no backend), validate, and the
# "no inbound ingress" policy for single-node. No AWS credentials or API calls are needed.
# Run: bash tools/ci_terraform.sh      Hardware: T0. Needs terraform >= 1.6 on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
terraform fmt -check -recursive infra
for d in infra/aws/single-node infra/aws/eks infra/aws/distributed infra/aws/failover infra/gcp/gke; do
  [ -d "$d" ] || continue
  echo "== $d"
  (cd "$d" && terraform init -backend=false -input=false >/dev/null && terraform validate)
done
if grep -REn --include='*.tf' '^[[:space:]]*ingress[[:space:]]*\{' infra/aws/single-node; then
  echo "single-node must not define ingress rules"; exit 1
fi
echo "terraform checks OK"
