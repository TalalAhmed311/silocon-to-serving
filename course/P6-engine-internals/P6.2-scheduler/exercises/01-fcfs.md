# Exercise 1: FCFS continuous batching (T0)

Implement `step()` and `update()` in `my_scheduler.py` **without** chunking or preemption first:

- running sequences each get their uncomputed tokens (1 for a decode);
- then admit waiting sequences in arrival order while the whole prompt fits in the remaining budget, `len(running) < max_num_seqs`, and `blocks.can_allocate(seq, n)`; call `blocks.match_prefix(seq)` before computing `n`;
- a `Chunk(seq, seq.num_computed, n)` per scheduled sequence; `update()` advances `num_computed`, commits blocks, appends sampled tokens, finishes and frees.

`test_fcfs_all_finish_within_budget` drives your scheduler with the fake model and checks every output equals the unbatched reference, the budget is never exceeded, and every block is free at the end.

**Then:** why must the FCFS loop `break` (not `continue`) when the head of the waiting queue doesn't fit?
