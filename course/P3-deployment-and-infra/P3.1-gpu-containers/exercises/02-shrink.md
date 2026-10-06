# Exercise 2 — Shrink the serving image ≥ 30% (T0, needs Docker)

1. Build `env/Dockerfile.serving` as written and record `docker image ls` (the SIZE column) and `docker history --no-trunc`.
2. Find the biggest layers and reduce them. Ideas:
   - `--no-cache` for pip/uv (already set: verify it took effect)
   - strip test directories and `__pycache__` from site-packages in the build stage
   - drop optional extras you don't serve with
   - check whether you're shipping two copies of the CUDA libs: the wheel's, plus a CUDA base image's
3. Record before and after in `results/p31.json`, as `{"before_bytes": ..., "after_bytes": ..., "changes": ["..."]}`, and run `uv run python course/P3-deployment-and-infra/P3.1-gpu-containers/exercises/check_p31.py results/p31.json`.
4. Make sure the slimmed image still serves (T2): run it on the GPU instance and send one request.
