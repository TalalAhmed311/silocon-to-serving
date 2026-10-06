#!/usr/bin/env bash
# Live NetworkPolicy probe (kind or EKS). Prints allowed/blocked for each path and exits 1 if any result is unexpected.
# Uses the platform image's Python (no curl in it); the stray pod runs with a restricted-PSS securityContext.
set -uo pipefail
NS=${NS:-s2s}
probe() {  # $1 = exec target (deploy/x or pod/x), $2 = url
  kubectl -n "$NS" exec "$1" -- python -c "import urllib.request,sys
try: urllib.request.urlopen('$2', timeout=3); print('allowed')
except urllib.error.HTTPError: print('allowed')
except Exception: print('blocked')" 2>/dev/null || echo blocked
}
kubectl -n "$NS" run np-stray --image=s2s-platform:dev --restart=Never --labels=app=stray \
  --overrides='{"spec":{"securityContext":{"runAsNonRoot":true,"runAsUser":10001,"seccompProfile":{"type":"RuntimeDefault"}},"containers":[{"name":"np-stray","image":"s2s-platform:dev","command":["sleep","300"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]}}}]}}' >/dev/null
kubectl -n "$NS" wait --for=condition=Ready pod/np-stray --timeout=60s >/dev/null
fail=0
check() { local got; got=$(probe "$2" "$3"); printf '%-34s %-8s (expect %s)\n' "$1" "$got" "$4"; [[ $got == "$4" ]] || fail=1; }
check "gateway -> mockllm"            deploy/gateway  http://mockllm.$NS:8001/health        allowed
check "stray pod -> mockllm"          pod/np-stray    http://mockllm.$NS:8001/health        blocked
check "stray pod -> gateway"          pod/np-stray    http://gateway.$NS:9000/health        blocked
check "mockllm -> IMDS"               deploy/mockllm  http://169.254.169.254/latest/meta-data/ blocked
check "mockllm -> gateway"            deploy/mockllm  http://gateway.$NS:9000/health        blocked
kubectl -n "$NS" delete pod np-stray --wait=false >/dev/null
exit $fail
