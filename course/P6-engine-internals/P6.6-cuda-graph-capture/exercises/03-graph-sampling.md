# Exercise 3 (hard): graph-safe sampling (T2)

Sampling after the graph means one more eager launch sequence and a device→host copy per step. Put it **inside** the graph:

- `sample_torch` uses `torch.multinomial` and a Python `if (top_k > 0).any()` — the `if` is a host sync. Rewrite it with no host syncs (always run every filter; a disabled filter is a no-op by construction, e.g. `top_k = V`).
- Randomness: a captured generator replays the *same* random numbers unless its state advances. Use PyTorch's graph-safe generator handling (register the generator with the graph; see the PyTorch CUDA graphs docs at your version) or pass per-row Philox offsets as a static input tensor you update before each replay.
- Check: greedy outputs unchanged; sampled outputs with a fixed seed reproducible across runs; the distribution test from P6.5 passes on samples drawn through replays.
