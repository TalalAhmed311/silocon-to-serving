# Exercise 2: resume with a different world size (T0, slow)

Losing a node often means resuming on **fewer** (or, after scaling up, more) ranks. DCP stores each tensor's global shape plus a map of which rank wrote which shard, so a load reshards to whatever layout the new process group has.

`test_reshard.py` (marked `slow`; run it with `uv run pytest -m slow …`):

1. world 2, 60 steps, uninterrupted → reference loss at step 59
2. world 2 for 40 steps (checkpoints at 19 and 39) → world **4** from step 40 to 59
3. requires `resumed_from == 39`, world 4, and the final loss within 0.2% of the reference

Why can they match at all? Read `train.batch()`: the **global** batch is fixed, and every sequence is generated from `(step, sequence index)`, never from the rank. Change it to seed by rank and watch the test fail. That's a real-world bug: resharding silently changes the data order.
