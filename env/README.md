# env/: reproducible environments

| File | Purpose | Tier |
|---|---|---|
| `../pyproject.toml` + `../uv.lock` | the course's Python env (`uv sync`, `uv sync --extra torch`, `--extra docs`) | T0 |
| `Dockerfile.cpu` | compilers, CMake, perf, and the locked Python env | T0 |
| `Dockerfile.cuda-dev` | CUDA toolkit (devel) + the torch extra, for Lane B and P5 | T2 |
| `Dockerfile.serving` | slim, non-root vLLM 0.31.0 serving image, no weights baked in | T2/T3 |
| `Dockerfile.platform` | gateway + mockllm for kind/K8s labs | T0 |
| `requirements-serving.txt` | vLLM + GuideLLM pins (GPU venv) | T2 |
| `requirements-sglang.txt` | SGLang pin (separate venv) | T2 |
| `requirements-quant.txt` | LLM Compressor, GPTQModel, lm-eval pins | T2 |

**Pin base images by digest** in your fork. Tags move; digests don't. The P3.1 exercise shows how, and the CI lint checks that every `FROM` has either a digest or an explicit version tag.
