# Stage 1 report — source index

**Date:** 2026-10-06. **Hardware used:** none (cloud sandbox, no GPU). **Network:** git over HTTPS to github.com, raw.githubusercontent.com and pypi.org were reachable. Every other website was blocked by the sandbox's egress policy.

## What was done

- **44 repos from the prompt, plus 2 found along the way** (`AlphaGPU/leetgpu-challenges`, `vvinjamu/silicon-to-scale`):
  - each pinned to its HEAD SHA
  - latest release tag recorded
  - license read at the pinned SHA
  - full file tree listed
  - results are in [SOURCES.md §1](../SOURCES.md)
- **Every repo path the prompt names** was checked against the pinned tree. See SOURCES.md §1.1.
- **The LeetGPU challenge list** came from its public source repo, because leetgpu.com was blocked. It has 107 challenges:
  - mapped to L1–L6: 90
  - electives: 17
- **The three anchor resources:**
  - Modular handbook: page index and interactive components read from its repo.
  - Silicon to Scale: chapter list and reading paths read from its repo's PDFs.
  - Inference Engineering Academy: **could not be verified.**

## Missing, moved or changed

| Item | Finding |
|---|---|
| SGLang `sgl-kernel/` | moved → `python/sglang/kernels/{jit,aot,ops}` |
| llama.cpp `ggml-quants.c` block formats | block structs now in `ggml/src/ggml-common.h`; per-arch SIMD in `ggml/src/ggml-cpu/arch/*/quants.c` |
| vLLM `benchmarks/` | still there, but the maintained CLI is `vllm bench` in `vllm/benchmarks/` |
| CUDA samples | moved from `Samples/` to `cpp/` |
| AutoAWQ | **deprecated**; its README says it was adopted by vLLM LLM Compressor, so #7 uses LLM Compressor |
| awsome-distributed-training | latest tag is `v2.0.1-pre-reorg`, so we pin by SHA and re-check paths before P4 |
| LeetGPU license | **CC BY-NC-ND 4.0**: we link only and never reproduce statements, starters or tests |
| LeetGPU GELU | no plain GELU challenge exists; GEGLU stands in (L4) |
| Silicon to Scale Fast-Track | the book's Path 2 chapters (6, 11, 8, 7, 9, 17, 14, 5) differ from the prompt's list (1, 6, 8, 9, 11, 13, 17); the syllabi cover both |
| Inference Engineering Academy | site blocked and no source repo found: **UNVERIFIED**, and nothing is built on it |
| Paywalled | CS:APP, C++ Concurrency in Action, PMPP, DDIA, K8s Up & Running, Terraform Up & Running, AI Engineering are books; chapters are assigned, and the learner needs their own copies |

## Open TODOs

See [GAPS.md](../GAPS.md).
