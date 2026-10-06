#!/usr/bin/env bash
# cost.sh — month-to-date spend for resources tagged Project=silicon-to-serving (needs ce:GetCostAndUsage).
set -euo pipefail
start=$(date -u +%Y-%m-01); end=$(date -u -d tomorrow +%Y-%m-%d 2>/dev/null || date -u -v+1d +%Y-%m-%d)
aws ce get-cost-and-usage --time-period Start=$start,End=$end --granularity MONTHLY --metrics UnblendedCost \
  --filter '{"Tags":{"Key":"Project","Values":["silicon-to-serving"]}}' \
  --query 'ResultsByTime[0].Total.UnblendedCost' --output table
echo "Running S2S instances:"
aws ec2 describe-instances --filters Name=tag:Project,Values=silicon-to-serving Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' --output table
