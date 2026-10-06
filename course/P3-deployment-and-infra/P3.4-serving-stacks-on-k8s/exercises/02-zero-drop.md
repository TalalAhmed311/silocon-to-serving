# Exercise 2 — Rolling update with zero dropped requests (T3, rehearse on kind)

1. Rehearse on kind with mockllm (P3.3 exercise 3).
2. On EKS, run `drain_test.py` from a pod inside the cluster against `http://vllm.s2s:8000` (with the API key header added), and `kubectl -n s2s rollout restart deploy/vllm` while it runs. With `maxSurge: 1` and a scale-from-zero GPU pool, the rollout waits for a new GPU node: record how long.
3. The test must exit 0. If it doesn't, find which mechanism failed: the preStop, the grace period, the readiness probe, or the surge waiting for a node. Fix it, then repeat.
