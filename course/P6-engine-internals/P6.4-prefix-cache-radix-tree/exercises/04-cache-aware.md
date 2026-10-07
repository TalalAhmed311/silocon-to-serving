# Exercise 4 (hard): cache-aware scheduling (T0)

Add a scheduling policy to your P6.2 scheduler (or a subclass of the reference): among waiting requests with equal effective priority, admit the one with the **longest cached prefix** first (compute it with `match_prefix` on a radix tree, or by probing block hashes without acquiring them).

Measure on the few-shot trace with a cache capacity of 25 % of the working set: total tokens computed (lower is better), mean and max waiting steps, versus FCFS. Then turn on aging and show the max wait is bounded again. Deliverable: a 4-row table (FCFS, LPM, LPM + aging, and your best variant) and two sentences on the trade-off.
