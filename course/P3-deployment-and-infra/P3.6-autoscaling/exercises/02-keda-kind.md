# Exercise 2: KEDA ScaledObject on kind with the mock backend (T0)

1. On your P3.3 kind cluster (mockllm deployed, kube-prometheus-stack from P3.5, `podmonitors.yaml` applied), install KEDA v2.21.0 and apply `platform/autoscaler/kind/scaledobject-mock.yaml`.
2. Confirm Prometheus has the signal: `sum(vllm:num_requests_running{namespace="s2s"})` returns a value in the Prometheus UI.
3. Record the timeline while driving load, then stop the load:

```bash
kubectl -n s2s get pods -w --output-watch-events | ts '%.s' | tee results/p3.6-kind.log &
kubectl -n s2s port-forward svc/mockllm 8001:8001 &
uv run python -m loadgen.cli --url http://localhost:8001 --rates 2 16 32 --duration 120 --out results/p3.6-kind-load.json
sleep 600; kill %1
python course/P3-deployment-and-infra/P3.6-autoscaling/exercises/check_scale_log.py results/p3.6-kind.log --deployment mockllm --expect-zero
```

**Pass:** the checker prints `OK`, with peak replicas ≥ 2, never above 4, and back to 0.

**Notice:** the port-forward targets the Service, so it breaks when the last pod goes away and has to be restarted after scale-from-zero. A real client goes through the gateway, which is the reason for the second trigger in the production `scaledobject.yaml`.

**Then:** set `cooldownPeriod: 30` and replay a bursty load (`--rates 16 0.1 16`). Count the cold starts. That's the flapping the policy in exercise 1 prevents.
