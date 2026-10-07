# P3.1 — GPU containers

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for building and inspecting images (Docker, no GPU); ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) to run the GPU images (`g4dn.xlarge` is enough) |
| **Time** | ≈25 min reading + ≈7 h hands-on |
| **Prerequisites** | P2.1 (you know what vLLM needs at runtime) |
| **You will build** | the course images in `env/`, and a Dockerfile policy linter |

## Learning objectives

1. Explain the layers: **host driver** (kernel module + `libcuda.so`, from the host) vs the **CUDA runtime/toolkit** inside the container, and NVIDIA's forward-compatibility rule.
2. Build slim serving images: multi-stage, pinned base digests, non-root, no weights baked in.
3. Use the NVIDIA Container Toolkit (`--gpus all`) and know what it injects.
4. Relate image size and pull time to **cold start** (P3.6).

---

## 1. What lives where

```
container image:  your app, Python, torch/vLLM wheels, CUDA *runtime* libs (libcudart, cuBLAS, NCCL, ...)  ← you control
-------------------------------------------------------------------------------------------- container boundary
host:             NVIDIA kernel driver + libcuda.so (the "driver API")  ← mounted in by the NVIDIA Container Toolkit
                  GPU device nodes (/dev/nvidia*)
```

- The **driver** comes from the host and is injected into the container at start (`nvidia-container-toolkit` / CDI). You never install a driver *inside* an image.
- The **CUDA runtime** in the image must not be newer than the driver supports. NVIDIA's rule of thumb: a driver supports CUDA runtimes up to its own CUDA version, and minor-version compatibility (11.x, 12.x) relaxes this within a major version for most features. Check NVIDIA's CUDA compatibility documentation for your exact versions. It is **UNVERIFIED** here, because the build environment couldn't reach docs.nvidia.com.
- `nvidia-smi` in a container shows the **host driver's** CUDA version. `nvcc --version` or `torch.version.cuda` show the image's. Record both (`examples/01_check_cuda_compat.sh`).

## 2. Serving images: what makes them good

| Practice | Why |
|---|---|
| multi-stage build (build venv → copy into a slim runtime) | no compilers, caches or headers in the final image, so smaller pulls and fewer CVEs |
| pin the base by **digest** (`FROM python:3.12-slim@sha256:…`) | tags move, so the "same" Dockerfile builds a different image next week |
| non-root `USER`, read-only root FS, caches on `/tmp` or a volume | defense in depth: a compromised server process can't modify its own image |
| **no weights in the image** | a 16 GB layer makes every pull slow and every model update a rebuild. Load at runtime (P3.7) |
| `HEALTHCHECK` / K8s probes with a long start period | vLLM takes minutes to load weights and capture graphs. Don't kill it while it's starting |
| one process per container | let K8s restart it, scale it and read its logs |

## 3. Image size is cold-start time

Pulling an image means downloading and decompressing every layer. A 10 GB vLLM image at a few hundred MB/s from a regional registry takes tens of seconds before Python even starts. Then come the weights (P3.7) and the CUDA graph capture (P2.6). P3.6 attacks all three. Here, measure the image part:

```bash
docker image ls s2s-vllm          # size
docker history s2s-vllm:0.31.0    # which layer is big?
time docker pull <registry>/s2s-vllm:0.31.0   # on a fresh node
```

> **Predict first.** Your image is 9 GB compressed. The node pulls at 400 MB/s and decompresses at 200 MB/s of compressed input. Estimate the pull + extract time. (≈ 9000/400 + 9000/200 ≈ 22 + 45 ≈ 67 s, if the two don't overlap.) Then measure it on a fresh `g6.xlarge` in P3.6.

---

## Walkthrough

```bash
docker build -f env/Dockerfile.platform -t s2s-platform:dev .              # T0
docker run --rm -p 8001:8001 s2s-platform:dev                              # mockllm in a container
docker build -f env/Dockerfile.serving -t s2s-vllm:0.31.0 .                # T0 to build (large download), T2 to run
docker run --rm --gpus all -v /opt/models:/models:ro -p 127.0.0.1:8000:8000 s2s-vllm:0.31.0 /models/<model> --host 0.0.0.0
bash course/P3-deployment-and-infra/P3.1-gpu-containers/examples/01_check_cuda_compat.sh s2s-vllm:0.31.0   # T2
uv run python course/P3-deployment-and-infra/P3.1-gpu-containers/examples/02_dockerfile_policy.py env/Dockerfile.*
```

## What you should see

- `02_dockerfile_policy.py` prints, per Dockerfile, the policy violations: unpinned base, root user, `ADD <url>`, secrets in `ENV`. On the files in `env/` it reports the base images that are pinned by tag but not by digest. That is intentional: exercise 1 asks you to pin them.
- `01_check_cuda_compat.sh` prints the host driver, the container's CUDA runtime, torch's CUDA build and the GPU name. `TODO(run-on: g4dn.xlarge)`.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Pin every base image by digest](exercises/01-pin-digests.md) | T0 | `test_policy.py`: `env/Dockerfile.*` pass the strict policy |
| 2 | [Shrink the serving image ≥ 30%](exercises/02-shrink.md) | T0 (Docker) | your before/after `docker image ls` in `results/p31.json`, checked by `check_p31.py` |
| 3 | [Non-root + read-only root FS](exercises/03-readonly.md) | T0 | `docker run --read-only --tmpfs /tmp …` serves requests (mockllm image) |
| 4 | [Reproducible build](exercises/04-reproducible.md) | T0 | two builds with `SOURCE_DATE_EPOCH` and BuildKit give the same image digest |

## Common mistakes

- Installing a driver inside the image. It conflicts with the host's injected `libcuda`.
- `FROM nvidia/cuda:latest`: unpinned, and huge (devel). Use `runtime` or a slim Python base for serving.
- Baking the HF token into an image layer. Layers are forever, even if a later layer deletes the file.
- `COPY . .` before `pip install`, which invalidates the dependency cache on every code change.

## Go deeper

- NVIDIA Container Toolkit docs and the CUDA compatibility guide (docs.nvidia.com: UNVERIFIED from the build environment).
- vLLM `docker/Dockerfile` and `docs/deployment/docker.md` at `vllm-project/vllm@d1f3d8b8`: how the official image is built.
- *Kubernetes Up & Running*, chapter on container images.

**Next:** [P3.2 Terraform on AWS](../P3.2-terraform-on-aws/README.md).
