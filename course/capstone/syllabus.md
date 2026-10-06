# Capstone — Failover and the public teardown

**Weeks 41–44 · Lane A ≈ 44 h · Tier T3 (two regions or two clouds) · Lane B: L6 daily, hard problems**

The capstone does not teach a new topic. It proves the platform works end to end, survives a regional failure, and can be reproduced by a stranger.

| Module | Time | Project |
|---|---|---|
| C1 Multi-region failover drill | 0.5 h + 20 h | **#14** |
| C2 Public benchmark teardown | 0.3 h + 22 h | **#15** |

## C1 — Multi-region failover drill (#14)

**Objectives.** (1) Active-passive vs active-active for stateful-ish LLM serving: what state actually matters (KV caches are disposable; tenant budgets, keys and audit logs are not). (2) DNS failover (Route 53 health checks / failover records, or the second cloud's equivalent), and gateway-level failover in #6. (3) Write RTO/RPO targets, then run a game day: kill a region and measure detection time, failover time, requests lost and budget-state divergence.

**Exercises.** (1) Failover decision logic as a pure function with tests (T0) · (2) health-check design that does not flap (hysteresis tests, T0) · (3) the game-day runbook and its measured timeline (T3) · (4) *(hard)* active-active with per-tenant budget reconciliation.

**Sources.** Google SRE books (managing incidents, handling overload) · *Designing Data-Intensive Applications* (replication chapter) · handbook `multi-cloud-and-cross-region-inference.md` · AWS Route 53 docs (UNVERIFIED).

## C2 — Public benchmark teardown (#15)

**Deliverable.** A reproducible report: architecture (Mermaid + the platform diagram), every component's version (SHAs from SOURCES.md), how to reproduce each number (`make bench-*`), latency/throughput/cost tables from #4/#2/#13, the custom-kernel delta (#12), #0 v1 vs vLLM, failover results (#14), and an honest "what didn't work" section.

**Acceptance.** A clean clone + the README reproduces the T0 numbers exactly, and the T2/T3 numbers within stated variance on the stated hardware. Every number links to a raw JSON result in the repo.

**Sources.** All prior phases. The Silicon to Scale ch 18 (principal career / system design) structure is used for the write-up and interview prep.
