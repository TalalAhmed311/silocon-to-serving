# Exercise 4 — Temperature + top-p sampling (medium)

Implement `ops::sample_top_p(logits, n, temperature, top_p, u)`:

1. If `temperature <= 0`, return the argmax.
2. Compute `p = softmax(logits / T)`.
3. Sort the token ids by `p` descending, breaking ties by id ascending, so the result never depends on `std::sort`'s internals.
4. Keep the smallest prefix whose cumulative mass `M ≥ top_p`.
5. Return the first kept token whose cumulative mass exceeds `u · M`. This samples the renormalized nucleus without dividing.

**Test:** `ctest -R 04`
- greedy at T = 0
- top-p 0.4 always returns the argmax
- **χ² test:** with p = {0.5, 0.3, 0.15, 0.05} and top-p 0.79, 10⁵ seeded draws must fit {0.625, 0.375, 0, 0} (χ² < 10.83, which is p = 0.001 with 1 degree of freedom)
- top-p 1.0 keeps the tail
- T = 0.25 sharpens

**Think:** why is the caller, not the function, responsible for the random number `u`?
