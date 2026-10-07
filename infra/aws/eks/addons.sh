#!/usr/bin/env bash
# addons.sh — install the cluster add-ons the course uses, with pinned chart versions.
# Run after `make up` and `aws eks update-kubeconfig ...`.
# Chart versions are UNVERIFIED pins (charts version independently of apps): check `helm search repo <chart> --versions`.
set -euo pipefail
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# NVIDIA GPU Operator v26.7.1 (SOURCES.md). Driver/toolkit come with the AL2023 NVIDIA AMI -> disabled here.
helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace \
  --version v26.7.1 -f ../../../platform/deploy/gpu-operator-values.yaml

# Prometheus + Grafana (kube-prometheus-stack). Version UNVERIFIED.
helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
  --version "${KPS_CHART_VERSION:-75.0.0}" -f ../../../platform/observability/kps-values.yaml

# KEDA (app v2.21.0 in SOURCES.md; the chart version usually matches the app version — verify).
helm upgrade --install keda kedacore/keda -n keda --create-namespace --version "${KEDA_CHART_VERSION:-2.21.0}"
kubectl get pods -A | grep -E 'gpu-operator|monitoring|keda'
