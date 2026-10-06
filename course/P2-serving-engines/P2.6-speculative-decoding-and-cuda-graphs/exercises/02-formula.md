# Exercise 2 — E[tokens] formula vs simulation (T0)

Write `simulate(alpha, k, rounds, seed) -> float`: the mean tokens per round when each proposal is accepted independently with probability α, the round stops at the first rejection (which still emits 1 corrected token), and a full acceptance adds the bonus token. Compare with `platform/specdec/core.expected_tokens_per_round`.

**Test:** `test_formula.py`, within 3% for α ∈ {0.3, 0.6, 0.9} and k ∈ {1, 4, 8} over 20k rounds. Then **think:** real acceptance is *not* i.i.d. (easy spans and hard spans cluster). Does clustering make the formula optimistic or pessimistic for a fixed mean α? Simulate a two-state (easy/hard) Markov acceptance process to find out, and write down the answer.
