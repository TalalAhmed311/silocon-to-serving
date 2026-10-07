# Exercise 3 — preStop drain: zero dropped streams during a rollout (T0, kind)

Run `drain_test.py` with a port-forward to the Service and `kubectl rollout restart deploy/mockllm` in the middle. It keeps 16 concurrent streams open (64 tokens each) for 60 s and counts truncated or failed streams.

1. With the shipped preStop (`sleep 5`) and a grace period: expect 0 errors. A `kubectl port-forward` pins one pod, so for a true test run the client **inside** the cluster: `kubectl run` a pod from the platform image that runs the script against `http://mockllm.s2s:8001`.
2. Set the preStop to nothing and `terminationGracePeriodSeconds: 1`, then repeat. Count the errors and explain each kind.
