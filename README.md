# Silicon-to-Serving

A self-paced course, about 44 weeks at ≈15 h/week, for one learner who wants to become an **AI infrastructure and inference engineer**: someone who can deploy and operate LLM serving on cloud GPUs and write CUDA C++ kernels.

The course has two parts:

1. A short **systems primer (P0)**.
2. A **top-down path** from serving and deployment down to CUDA kernels and engine internals (P1–P6), then a capstone.

Two lanes run in parallel:

| Lane | What | Pace |
|---|---|---|
| **A** | Phases + portfolio projects | ≈11 h/week |
| **B** | CUDA C++ kernels on [LeetGPU](https://leetgpu.com/challenges) | 3–4 problems/week in P0–P4, daily from P5 |

> **Build status: Stages 1–4 done; Stage 5 (QA) is a first pass.** Every phase P0–P6, the Capstone and Lane B L1–L6 are written: lessons, examples, exercises with tests, benches, animations, AWS/GCP guides, and the `platform/` projects. **Nothing has been executed yet**: T0 tests and examples are `TODO(run)`, hardware runs are `TODO(run-on: …)`. [GAPS.md](GAPS.md) lists every one, with the commands to close them.

## Course map

| Phase | Weeks | Topic | Syllabus | Projects |
|---|---|---|---|---|
| P0 | 1–3 | Systems primer: C/C++, OS, memory, SIMD | [syllabus](course/P0-systems-primer/syllabus.md) | #0 v0, D1, D2 |
| P1 | 4–6 | How LLM inference works + hardware math | [syllabus](course/P1-inference-fundamentals/syllabus.md) | D3 |
| P2 | 7–12 | Serving with inference engines | [syllabus](course/P2-serving-engines/syllabus.md) | #4, #7, #10, #6 |
| P3 | 13–20 | Deployment and infra on multiple clouds | [syllabus](course/P3-deployment-and-infra/syllabus.md) | #1, #13, #2, #3, #5, #8 |
| P4 | 21–24 | Multi-GPU, GPU sharing, distributed jobs | [syllabus](course/P4-multi-gpu-and-distributed/syllabus.md) | #9, #11 |
| P5 | 25–34 | CUDA C++ deep dive | [syllabus](course/P5-cuda-deep-dive/syllabus.md) | D4, #12 |
| P6 | 35–40 | Engine internals | [syllabus](course/P6-engine-internals/syllabus.md) | #0 v1 |
| Cap | 41–44 | Multi-region failover + public teardown | [syllabus](course/capstone/syllabus.md) | #14, #15 |
| Lane B | 1–44 | CUDA levels L1–L6 mapped to LeetGPU | [overview](laneB-cuda/README.md) | — |

## Hardware tiers

Every module and example carries one of these tiers:

| Tier | Where | Used for |
|---|---|---|
| **T0** | Laptop, no GPU (Linux/macOS, x86 or ARM) | P0, P1 calculators, gateway logic, every unit test, Triton interpreter mode |
| **T1** | LeetGPU in the browser | Lane B L1–L4 |
| **T2** | One local or rented NVIDIA GPU (AWS `g4dn`/`g5`/`g6`) | P2 single-node serving, P5 kernels + Nsight |
| **T3** | AWS multi-GPU / multi-node | P3 cluster, P4 distributed, MIG, failover |

**Cloud safety rules for every AWS guide:**

- the hourly cost is stated
- an idle auto-stop is set up
- a budget alarm is set up
- `make down` is tested
- access is SSM-only (no open SSH)
- no unauthenticated public endpoint

## Repository layout

```
README.md         this file
SOURCES.md        every external source: URL, pinned SHA/version, license, status, modules that use it
GLOSSARY.md       one-line definitions of every term
GAPS.md           everything UNVERIFIED or TODO(run-on: …), with the exact command to close it
PROGRESS.md       week-by-week checklist
course/           P0–P6 + capstone; each phase has syllabus.md and one folder per module
laneB-cuda/       L1–L6; each has syllabus.md and leetgpu-map.md; plus harness/ and electives.md
platform/         the single portfolio repo that the projects grow into
animations/       source of every animation (self-contained HTML/JS/SVG)
site/             MkDocs Material config that renders the course
env/              Dockerfiles and lockfiles (cpu, cuda-dev, serving)
infra/aws/        Terraform + scripts (single-node, EKS, distributed, failover, guardrails); infra/gcp/gke for P3.9
tools/            link checker, bench runner, notebook executor
reports/          stage and phase reports
```

## Where things stand

| Stage | Status |
|---|---|
| 1. Source index | **Done.** See [SOURCES.md](SOURCES.md) and [reports/stage1-source-index.md](reports/stage1-source-index.md) |
| 2. Syllabus | **Done.** Approved with defaults: book Path 2 order, GKE as the second cloud, Ray on EKS for P4.3 |
| 3. Pilot module (P0.1, end to end) | **Done** |
| 4. Build-out (P0 → Capstone, Lane B alongside) | **Done:** P0–P6, Capstone (#14, #15), L1–L6 solutions and hint pages, D4 kernels, #0 v1 engine, 20 animations |
| 5. QA and delivery | **First pass done:** links checked (`tools/check_links.py`), Python and YAML parse-checked, GAPS/GLOSSARY/PROGRESS regenerated. **Open:** the first real `uv run pytest` and every hardware run (GAPS §C–D) |

## Quickstart

```bash
uv sync                                    # Python ≥ 3.11; installs the T0 stack
S2S_SOLUTIONS=1 uv run pytest              # every T0 test against the reference solutions (first run: GAPS.md §C)
uv run pytest course/P0-systems-primer     # your exercise stubs: they fail until you implement them
S2S_RUNNER=fake uv run uvicorn s2s_engine.server:app --app-dir platform/engine/v1 --port 8002   # the #0 v1 engine, fake model
python tools/build_site.py serve           # the course as a website (needs site/requirements.txt)
```

Start with [P0.1](course/P0-systems-primer/P0.1-c-cpp-for-systems/README.md) and tick weeks off in [PROGRESS.md](PROGRESS.md).

## License

The course text is original. External material is linked and summarized, never copied. LeetGPU problems are CC BY-NC-ND 4.0, so only links appear here. See [SOURCES.md](SOURCES.md) for every source's license.
