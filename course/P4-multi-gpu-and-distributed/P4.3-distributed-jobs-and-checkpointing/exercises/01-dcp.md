# Exercise 1: DCP round trip, bitwise-equal state (T0)

1. `test_dcp.py` saves #9's `TinyLM` and its AdamW state with `torch.distributed.checkpoint`, loads them into a model with different values, and requires **bitwise** equality (model and optimizer: `exp_avg`, `exp_avg_sq`, `step`). Run it, then read `examples/02_dcp_save_load.py` and look at the files DCP wrote.
2. `test_checkpoint_logic.py` covers what DCP doesn't: the `COMMITTED` marker (a checkpoint without it is ignored on resume), latest-checkpoint discovery, and pruning that keeps an in-progress save. Explain why rank 0 writes the marker only **after a barrier**.
3. **Break it:** comment out the barrier in `checkpoint.save` and argue about what can now go wrong on a real cluster. Hint: rank 0 finishes first, writes `COMMITTED`, and then rank 3's node dies before its shard is flushed.
