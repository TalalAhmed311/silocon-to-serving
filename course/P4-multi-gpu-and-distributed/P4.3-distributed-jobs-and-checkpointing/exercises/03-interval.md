# Exercise 3: the optimal checkpoint interval (T0)

Checkpointing too often wastes time saving. Too rarely, and failures waste recomputation. Young's approximation `τ* = √(2·C·M)` balances them (C = checkpoint cost, M = mean time between failures). Daly refines it.

1. `test_interval.py` checks the formula, finds the minimum of the first-order waste model, and runs `faults.simulate` (exponential failures, restart cost R) at 0.25× … 4× τ*. Make sure you understand why both extremes lose.
2. **Plug in real numbers.** For #9 on T3, measure C (save time from the logs). Estimate M for a 16-GPU job if each GPU node has an MTBF of, say, 30 days (a placeholder: look for published fleet numbers, e.g. the Llama 3 paper's interruption statistics). What τ* does that give? And for 1,024 GPUs?
3. **Async checkpointing** (exercise 5) shrinks the *blocking* part of C. How does that move τ*?
