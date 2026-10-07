# Exercise 1: ring all-reduce simulator (T0)

Implement `reduce_scatter`, `all_gather` and `all_reduce` in `ring.py`. Ranks are a list of NumPy arrays. In each **step**, every rank sends one chunk to its right neighbour at the same time. Copy the outgoing chunk before applying any receive: messages are simultaneous.

`test_ring.py` checks:

- `all_reduce` equals `sum` on every rank, for n = 1…8 and sizes that don't divide evenly (`np.array_split`)
- `reduce_scatter` leaves rank r with the full sum of chunk r, in n−1 steps
- `all_gather` produces rank order, in n−1 steps
- all-reduce takes exactly **2(n−1)** steps

**Then:** count bytes. Each rank sends `2(n−1)/n · S` in total. Why is that independent of n for large n, and why does it make ring all-reduce **bandwidth-optimal**?
