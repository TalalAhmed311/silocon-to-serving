# Exercise 1 — The verification rule preserves the target distribution (T0)

Implement `verify(p, q, proposals, rng) -> list[int]` in [`verify.py`](verify.py). `p` is `[k+1, V]` (target), `q` is `[k, V]` (draft) and `proposals` is a length-k list. It applies the acceptance rule and returns the accepted prefix plus exactly one more token (the corrected one or the bonus).

**Test (`test_verify.py`):**
- over 20,000 rounds on random distributions, the empirical distribution of the **first** emitted token matches `p[0]` (χ², p > 0.001)
- the distribution of the **second** emitted token, conditioned on the first being the accepted proposal, matches `p[1]`
- a planted-bug variant (`verify_buggy`, which resamples from `p`) **fails** the same test. This is a test of the test
- with p == q, every proposal is accepted
