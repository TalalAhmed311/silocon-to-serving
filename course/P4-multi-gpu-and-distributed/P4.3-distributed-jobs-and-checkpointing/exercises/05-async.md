# Exercise 5 (hard): async checkpointing and its overhead

`dcp.save` blocks training while it serializes and writes. `torch.distributed.checkpoint.async_save` (PyTorch docs, UNVERIFIED for your version) copies the state to CPU memory and writes in a background thread, so training only blocks for the device→host copy.

1. Add `--async-ckpt` to `train.py`. Keep **one** in-flight save: wait on the previous future before starting the next. Write the `COMMITTED` marker only after the future completes **and** all ranks agree. That needs a collective, so do it at the next step boundary.
2. Measure the per-step time around a checkpoint with sync vs async on T3: `| mode | blocking time per checkpoint (ms) | extra host memory (GB) |`.
3. Recompute τ* (exercise 3) with the new blocking C.
4. Failure semantics: the job dies while an async save is in flight. Show (with `--die-at-step` right after a checkpoint step) that the resume uses the **previous** committed checkpoint, not the partial one.
