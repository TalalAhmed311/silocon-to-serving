# Exercise 2 — Readiness gates on warm-up (T0, kind)

Simulate a slow model load:

1. Add `MOCKLLM_STARTUP_DELAY_S`, an environment variable to `platform/mockllm/server.py` that makes `/health` return 503 until N seconds after start (use the FastAPI lifespan or a start timestamp). Write a unit test for it.
2. Deploy with `MOCKLLM_STARTUP_DELAY_S=30`. Watch `kubectl -n s2s get endpoints mockllm -w`: the pod IP must appear only after ~30 s.
3. Now remove the `readinessProbe`, redeploy, and send traffic immediately. Explain the errors you see. Then put the probe back.
