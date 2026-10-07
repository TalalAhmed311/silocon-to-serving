# Exercise 3 (hard): one scheduling difference, reproduced with a trace

Find one behavioural difference between nano-vllm's scheduler and vLLM v1's. Candidates to check (verify each at the pinned versions — don't take this list on trust):

- whether prefill and decode share a step (mixed batches / chunked prefill)
- which running request is preempted when blocks run out, and whether preemption is recompute or swap
- whether a request whose prompt doesn't fit the token budget is chunked, skipped, or blocks the queue

Write a **request trace** (arrival step, prompt length, max_tokens) that makes the difference visible, and show the per-step schedule from both. For a T0 version, reproduce both policies with #0 v1 (`SchedulerConfig(chunked_prefill=False)` approximates the "separate prefill" policy) and print traces with `examples/01_trace_one_step.py` adapted to your trace.

Deliverable: the trace, both step-by-step schedules, and two sentences on which policy wins TTFT and which wins ITL for that trace.
