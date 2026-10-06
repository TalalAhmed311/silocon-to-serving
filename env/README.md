# env/

These images are built in Stage 3/4. They will be pinned by base-image digest and lockfile.

| File | Purpose | Tier |
|---|---|---|
| `Dockerfile.cpu` | gcc/clang, CMake, Python + uv, perf | T0 |
| `Dockerfile.cuda-dev` | CUDA 12.4+ toolkit, Nsight CLI, CMake | T2 |
| `Dockerfile.serving` | pinned vLLM `v0.31.0` / SGLang `v0.5.21` runtime | T2/T3 |
| `uv.lock` / `requirements*.txt` | pinned Python deps per environment | all |
