# Exercise 4 (hard): priorities with an anti-starvation bound (T0)

Add `effective_priority(seq) = seq.priority + (seq.waited_steps // cfg.aging_steps if cfg.aging_steps else 0)`. Admit by highest effective priority, ties broken by arrival. Increment `waited_steps` for every sequence still waiting at the end of a step; reset it on admission and on preemption. Use effective priority for victim choice too.

Tests: `test_priority_strict_starves` (without aging, a priority-0 request never runs under a stream of priority-1 requests) and `test_priority_aging_bound` (with aging, it finishes within `2·aging_steps + 1` steps).

**Then:** prove the bound in general: a request of priority p waits at most `aging_steps · (p_max − p) + c` steps when `max_num_seqs = 1` and every request takes one step. What's `c`, and what changes when requests take longer than one step?
