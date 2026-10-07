# Exercise 4 — Idle auto-stop, tested with synthetic metrics (T0)

The single-node watchdog (`user_data.sh.tftpl`) has its decision logic mirrored as a pure function in `infra/aws/scripts/idle_decider.py`. `test_idle.py` drives it with synthetic minute-by-minute traces:

- busy, then idle for exactly `idle_limit` minutes → stops at that minute, not before
- idle, but with an SSM session open → never stops
- `nvidia-smi` failing (`None`) → counts as idle (a dead GPU shouldn't keep billing)
- a single busy minute resets the count

**Extend it:** add a "grace period after boot" (don't stop in the first 15 minutes after start, so you can SSM in), in both the Python mirror and the bash template, with a test. Keep the two in sync. That is the reason the mirror exists.
