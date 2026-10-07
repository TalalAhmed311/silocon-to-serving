# Stages 4–5 build report

## What was built

| Area | Contents |
|---|---|
| P0–P6 | every module's README, examples, exercises (starters + solutions + tests), benches, quizzes, and `aws.md` where a cloud GPU is needed |
| Capstone | C1 failover (#14) and C2 teardown (#15), with runbook and report builder |
| Lane B | L1–L6 syllabi, LeetGPU maps, reference solutions and hint pages, the local harness |
| platform/ | every project layer: capacity, mockllm, loadgen, bakeoff, specdec, gateway, deploy, observability, cost, autoscaler, weights, tenancy, training, partitioning, kernels (D4), engine v0 + v1, failover, report |
| infra/ | AWS single-node, EKS, distributed, failover, guardrails; GCP GKE |
| animations/ | 20 self-contained HTML animations |

## How it was checked

The build ran under an explicit instruction **not to run or download anything**. Checks were static only:

- `tools/check_links.py`: every relative Markdown link resolves.
- Python files parsed with `ast`; YAML parsed with PyYAML.
- Code reviewed by hand against its tests (several bugs found and fixed that way, e.g. a scheduler livelock where a waiting request held prefix-hit blocks, a server token-delivery race, a CUDA-graph padding hazard handled by reserving block 0).

Nothing has been executed: no tests, no examples, no Terraform, no animations opened. [GAPS.md](../GAPS.md) §C–E is the complete list of what must run, on which hardware, and what is unverified.

## Known risks for the first run

- Tests that were never executed will have some failures (typos, API drift in pinned libraries). Fix forward; run them locally with `S2S_SOLUTIONS=1 uv run pytest`.
- CUDA code was reviewed but never compiled: expect compile errors on the first `cmake --build`, especially around architecture-specific intrinsics (P5.7–P5.8).
- Version pins (vLLM `v0.31.0`, SGLang `v0.5.21`, provider `~> 6.0`, etc.) come from Stage 1's source index; re-check them against the current releases before the first T2/T3 run.
