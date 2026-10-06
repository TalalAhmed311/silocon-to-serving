# Exercise 2 — FLOPs per token (easy)

Implement `decode_flops(cfg, t) -> int` in `flops.py`: the FLOPs of one decode step at context length `t` (the new token attends to `t` positions, including itself). Count:

- `2 ×` every matrix weight used: q, k, v, o, gate, up, down and the LM head. Not the embedding: it is a lookup.
- Attention: `4 · H · h · t` per layer.
- **Ignore** norms, RoPE and activations. The test allows for them with a 3% tolerance: they are ~2% of the FLOPs of the d = 64 test model, and < 0.1% of a real one.

**Test:** compared against the instrumented counter from example 01 on the tiny model at t = 1, 10 and 100, `rtol = 3e-2`.
