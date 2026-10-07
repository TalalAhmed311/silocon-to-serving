# C1: Multi-region failover drill (#14)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the failover logic, hysteresis, budget reconciliation and timeline analysis (`platform/failover`, fully tested). ![T3](https://img.shields.io/badge/tier-T3%20two%20regions-red) for the game day |
| **Time** | ≈30 min reading + ≈20 h hands-on |
| **Prerequisites** | P3 (EKS platform, gateway tenancy and audit, observability), P2.7 (#6 gateway retries/fallbacks), P4.3 (RTO/RPO for training) |
| **You will build** | **#14**: a tested failover controller (`platform/failover`), Route 53 failover driven by an authenticated in-region prober (`infra/aws/failover`), and a measured game-day timeline |

## Learning objectives

1. Decide what state must survive a region loss in LLM serving, and what is safely disposable.
2. Choose active-passive or active-active, and state the RTO/RPO each can meet.
3. Build a health signal that detects real outages fast without flapping (hysteresis), and prove it with tests.
4. Wire DNS failover so that no unauthenticated health endpoint faces the internet.
5. Run a game day: kill a region, measure detection, failover time, requests lost and budget divergence, and write the runbook from what actually happened.

---

## 1. What state matters

| state | lives in | if the region dies | needed in the other region? |
|---|---|---|---|
| KV caches, prefix caches | GPU memory | gone | **no** — recomputed (a TTFT spike on the first requests) |
| model weights | S3 + node NVMe cache (P3.7) | the cache is gone | yes — **pre-stage** them in the secondary's bucket and warm a minimum of replicas, or cold start dominates RTO |
| in-flight requests | client connections | lost | no — clients retry (the gateway's retry budget, P2.7) |
| tenant keys | Secrets Manager (P3.8) | — | yes — replicate the secret to the secondary region |
| per-tenant token budgets | gateway (P3.8) | the region's counters stop | yes — or a tenant can overspend; §4 |
| audit log (hash chain) | S3 (P3.8) | the tail not yet shipped | yes — ship often; the RPO of the audit log is your shipping interval |

The lesson: an LLM serving region is mostly **stateless compute on expensive, slow-to-start hardware**. The hard part of failover is capacity (GPUs available in the other region, weights already there), not data.

## 2. Topologies and RTO/RPO

| | active-passive | active-active |
|---|---|---|
| traffic | one region; the other is warm standby | both regions serve |
| standby cost | the secondary's warm minimum (GPUs idle) | none idle, but each region needs headroom for the other's load |
| RTO | detection + DNS TTL + client resolver caching + secondary scale-up | detection + DNS (the survivor already serves) — plus scale-up for the doubled load |
| budget state | single writer, simple | concurrent writers: CRDT counters + allowance splitting (§4) |

**DNS is not instant.** Route 53 changes the answer once a health check flips, but clients and resolvers cache for the TTL — some longer. Measure "kill → first success in the new region" from a client *outside* both regions; that's the RTO users feel. The gateway can do better than DNS for its own backends (P2.7 fallbacks), so in practice: DNS for region failover, gateway fallback for backend failover.

## 3. A health signal that doesn't lie

`platform/failover/health.py`: a region is DOWN after `fail_threshold` consecutive failed probes and UP again only after `rise_threshold` consecutive good ones; a probe slower than the latency SLO counts as failed. Detection time ≈ `fail_threshold × interval`; flap rate falls steeply as the threshold rises (tests bound it under 20 % random loss).

**Authenticated probing.** Route 53's own HTTP health checkers come from public IPs and would need an endpoint open to them. Instead, `prober.py` runs *inside* each region, sends a real 1-token completion through the gateway with a dedicated probe tenant key, and publishes `RegionUp` to CloudWatch. A CloudWatch alarm per region feeds a Route 53 **CloudWatch-metric health check** (`infra/aws/failover`). Missing data counts as breaching: if the prober dies, you can't see the region, so treat it as down.

`platform/failover/policy.py` is the decision as a pure function: fail over only if the other region is up; hold if both are down; fail back only after the primary has been stable for `failback_after_s` (or never automatically); never switch twice within `min_dwell_s`.

## 4. Budgets in active-active

Each region counts tenant tokens in a **G-counter** (one grow-only counter per (tenant, region), merged by element-wise max) — merge order doesn't matter, so replicas converge. But between syncs each region sees only its own spend: a naive "used + n ≤ budget" check lets every region spend the whole remainder at once (`test_naive_budget_can_overspend`). `RegionBudget(split=True)` gives each region `remaining / regions` per sync interval, so the total can never exceed the budget (`test_split_budget_never_overspends`) — at the price of refusing early when traffic is skewed. Exercise 4 quantifies that.

## Walkthrough

```bash
uv run pytest platform/failover -q                                    # T0
# T3, see infra/aws/failover/README.md: two EKS regions + platform/deploy/overlays/eks-failover + the Route 53 stack
```

## What you should see

T0: all tests pass. T3: a timeline like (yours will differ — **measure it**): kill at T; alarm in ≈ `alarm_periods` minutes; health check flips shortly after; DNS answers change; first success in the secondary; steady state. `gameday.py` turns the client log into the table below.

## Bench (game day)

| metric | target (write it before the drill) | measured |
|---|---|---|
| detection (kill → health check unhealthy) | | `TODO(run-on: AWS, two regions)` |
| first success in secondary (client-side) | | |
| failover (20 consecutive successes in secondary) | | |
| requests failed / error rate during drill | | |
| budget overspend (active-active) | 0 | |
| audit-log RPO (last shipped entry before kill) | | |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Failover decision logic as a pure function](exercises/01-decide.md) | T0 | `test_failover.py::test_decide_table`, `test_controller_no_ping_pong` |
| 2 | [A health check that doesn't flap](exercises/02-hysteresis.md) | T0 | `test_failover.py::test_*flap*`, `test_outage_*`, `test_slow_is_down` |
| 3 | [The game day: runbook and measured timeline](exercises/03-game-day.md) | T3 | `gameday.py` + [runbook template](runbook.md) |
| 4 | *(hard)* [Active-active with per-tenant budget reconciliation](exercises/04-active-active.md) | T0 → T3 | `test_failover.py::test_*budget*`, `test_gcounter_*` |

## Common mistakes

- Measuring failover from inside the surviving region. Users are outside both.
- A health endpoint that returns 200 while the engines behind it are dead. Probe a real (tiny) completion.
- Forgetting the secondary's capacity: failover to a region with zero warm GPUs turns a 2-minute RTO into a cold start plus a GPU-availability lottery.
- Low TTLs and no client retries: DNS moved, but the client's pooled connection still points at the dead region.
- Leaving Route 53 health checks behind after the drill: they bill monthly.

## Go deeper

- Google SRE book: *Managing Incidents*, *Handling Overload*; SRE workbook: *Incident Response*.
- Kleppmann, *Designing Data-Intensive Applications*, ch. 5 (replication) — and Shapiro et al., *Conflict-free Replicated Data Types* for the G-counter.
- AWS Route 53 Developer Guide: failover routing, health checks based on CloudWatch alarms (UNVERIFIED wording — check the current docs).
- Modular handbook `multi-cloud-and-cross-region-inference.md`.

**Next:** [C2 Public benchmark teardown](../C2-public-benchmark-teardown/README.md).
