#!/usr/bin/env bash
# Install the KubeRay operator and the namespace/ServiceAccount the RayJob uses. Chart version UNVERIFIED: check
# `helm search repo kuberay/kuberay-operator --versions` and pick the one documented for Ray 2.59.0.
set -euo pipefail
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ && helm repo update
helm upgrade --install kuberay-operator kuberay/kuberay-operator -n kuberay --create-namespace \
  --version "${KUBERAY_CHART_VERSION:?set KUBERAY_CHART_VERSION}"
kubectl create namespace s2s-train --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace s2s-train pod-security.kubernetes.io/enforce=baseline --overwrite
kubectl -n s2s-train create serviceaccount s2s-train --dry-run=client -o yaml | kubectl apply -f -
kubectl -n kuberay get pods
