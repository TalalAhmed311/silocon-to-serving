# P6.2: The scheduler

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) everything: the scheduler is pure Python, tested with a fake model |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P6.1, P2.3 (continuous batching from the outside), P1.3 (TTFT/ITL) |
| **You will build** | a continuous-batching scheduler with chunked prefill, preemption by recompute and priorities with an anti-starvation bound — the one in `platform/engine/v1/s2s_engine/scheduler.py` |

## Learning objectives

1. Write the per-step loop: running sequences first, then admissions, inside a **token budget** and a **sequence cap**.
2. Explain chunked prefill as "schedule `min(remaining, budget)` tokens" and why it protects ITL.
3. Implement preemption by **recompute**, pick a victim, and argue why it can't livelock.
4. Add priority classes, show they can starve low-priority work, and bound the wait with aging.
5. Test all of it with a fake model whose output proves scheduling never changes results.

---

## 1. One step

The animation [`p6-scheduler-loop.html`](../../../animations/p6-scheduler-loop.html) runs this loop on five requests.

```
budget = max_num_batched_tokens
for seq in running (admission order):               # 1. keep running work going
    n = min(seq.num_uncomputed, budget)              #    decode: 1 token; unfinished prefill: a chunk
    while blocks can't hold n more tokens of seq:
        preempt the lowest-priority, newest running seq (not one already in this batch) — or seq itself
    allocate; schedule (seq, start=num_computed, n); budget -= n
if nothing was preempted:                            # 2. admit (never preempts running work)
    while budget and len(running) < max_num_seqs:
        seq = highest effective priority in waiting
        reuse cached prefix blocks (P6.3); n = min(seq.num_uncomputed, budget)
        if blocks can't hold it: stop
        allocate; move to running; schedule; budget -= n
```

Everything hinges on one counter per sequence, **`num_computed`**: tokens whose K/V are already in the cache. A decode is "1 uncomputed token". A prefill chunk is "the next n uncomputed tokens". A prefix-cache hit starts `num_computed` above zero. A preemption by recompute sets it back to zero. A chunk **produces a token** only if it reaches the end of the sequence's known tokens.

## 2. Chunked prefill

Without chunking, a 4,000-token prompt either waits for an empty step or monopolises one — every running decode stalls for that step and ITL spikes. With chunking, the prompt is fed in budget-sized pieces *alongside* the decodes (Sarathi-Serve's "stall-free batching"). The cost: the prompt's TTFT spans several steps. The knob is `max_num_batched_tokens`: smaller → smoother ITL, longer TTFT for long prompts.

> **Predict first.** Budget 512, 32 running decodes, one new 2,000-token prompt. How many steps until its first token? What's the extra per-step work for the decodes compared with no prompt at all? Check with `bench/budget_sweep.py`.

## 3. Preemption: recompute vs swap

When a running sequence needs a new block and none is free, something has to give:

| | recompute (ours, vLLM v1 default) | swap |
|---|---|---|
| on preempt | free the blocks; `num_computed = 0` | copy the blocks to CPU memory |
| on resume | re-prefill prompt + generated tokens | copy back |
| costs | prefill FLOPs (compute-bound, fast on GPU) | PCIe traffic both ways + host memory |

**Victim choice** matters for liveness. Ours: lowest effective priority, then most recently admitted, never a sequence already placed in this step's batch; if there's no other candidate, the sequence preempts *itself*. Admission is skipped in a step that preempted, so the freed memory goes to the oldest work first instead of being re-taken by a newcomer (thrash). Together with **rejecting at `add_request` any request that can never fit** (prompt + max_tokens > pool), the oldest running sequence always eventually gets the whole pool, so the system can't livelock. One more trap we hit while writing the reference: a waiting request that *holds prefix-hit blocks* while it waits can pin memory the running sequence needs — the scheduler releases them if admission fails.

## 4. Priorities and starvation

Strict priority starves: a stream of priority-1 requests means a priority-0 request never runs (`test_priorities_starve_without_aging`). **Aging** adds `waited_steps // aging_steps` to the effective priority, so after `aging_steps × Δpriority` waiting steps the old request ties with any fresh arrival and wins the tie on arrival time. That is a **provable bound**, which is the point of exercise 4.

## Walkthrough

```bash
uv run pytest platform/engine/v1/tests/test_scheduler.py -q                 # the reference passes
uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q        # your scheduler (fails until you implement it)
S2S_SOLUTIONS=1 uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q   # the tests against the reference
uv run python course/P6-engine-internals/P6.2-scheduler/bench/budget_sweep.py
```

## What you should see

- The reference tests pass, including `test_memory_pressure_actually_preempts` (preemptions > 0 and identical outputs).
- `budget_sweep.py` prints, per budget, steps to the long prompt's first token and the max tokens in any step. Smaller budget → more steps to first token, never a step above the budget.

## Bench

`budget_sweep.py` is a step-count model (fake model, T0). Real TTFT/ITL vs `max_num_batched_tokens` comes in P6.7 on a GPU: `TODO(run-on: g6.xlarge)`.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [FCFS continuous batching](exercises/01-fcfs.md) | T0 | `test_my_scheduler.py::test_fcfs_*` |
| 2 | [Chunked prefill](exercises/02-chunked.md) | T0 | `test_my_scheduler.py::test_chunked_*` |
| 3 | [Preemption by recompute](exercises/03-preempt.md) | T0 | `test_my_scheduler.py::test_preempt_*` |
| 4 | *(hard)* [Priorities with an anti-starvation bound](exercises/04-priority.md) | T0 | `test_my_scheduler.py::test_priority_*` |

## Common mistakes

- Advancing `num_computed` at schedule time instead of after the forward pass — then a failed step corrupts state.
- Sampling a token for a prefill chunk that doesn't reach the end of the prompt.
- Letting admission preempt running work: throughput collapses into thrash.
- Picking the victim among sequences already in this step's batch: you free blocks the batch is about to write.

## Go deeper

- Orca (OSDI '22): iteration-level scheduling. Sarathi-Serve (OSDI '24): chunked prefill and stall-free batching.
- vLLM `vllm/v1/core/sched/scheduler.py` at `v0.31.0`: find the token budget, the running-first loop and the preemption path (verify at the pin).
- nano-vllm `scheduler.py`: the same ideas without chunking.

**Next:** [P6.3 Paged KV block manager](../P6.3-paged-kv-block-manager/README.md).
