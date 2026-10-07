# Exercise 2: copy-on-write on fork (T0)

A forked child shares every block. When either sequence appends into a **partial** last block whose ref count is > 1, `allocate` must: take a new block, queue `(src, dst)` in `copy_ops`, release one reference to `src`, and put `dst` in the table. `can_allocate` must count the extra block.

`test_cow_after_fork` checks the queued copy, that the full first block stays shared, and the ref counts.

**Then:** in `model_runner.py`, find where each runner executes `copy_ops`. Why must it happen before the forward pass of the same step, and what would a parallel-sampling request with n = 4 cost in copies at its first decode?
