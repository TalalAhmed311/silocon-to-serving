# Exercise 1: capture and replay (T2)

Run `examples/01_capture_replay.py`, then break it on purpose, one rule at a time, and record what happens:

1. call `g.replay()` after `static_in = torch.randn(...)` (rebinding instead of `copy_`);
2. add `if x.sum().item() > 0:` inside `step`;
3. capture without the warm-up;
4. make `step` allocate a tensor whose size depends on an input value.

For each: does capture fail, does replay silently compute the wrong thing, or does it work? Explain why from README §2. Then vary the matrix size (64 → 4096) and find where replay stops saving meaningful time.
