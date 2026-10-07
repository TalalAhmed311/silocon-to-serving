# Exercise 1: choose a MIG layout for a tenant mix (T0)

Implement `plan(gpu, tenants)` in `layout.py`. For each tenant, pick the **smallest** MIG profile with enough memory and at least `min_compute` compute slices. Then bin-pack the profiles onto as few GPUs as possible, without exceeding 7 compute slices or 8 memory slices per GPU.

`test_layout.py` checks placement, slice limits, the 7 × `1g.10gb` case, hitting the lower bound on a mix, `min_compute`, and an impossible tenant. `S2S_SOLUTIONS=1` runs the reference (`platform/partitioning/mig.py`).

**Then:**

1. The real hardware also restricts *where* each profile can start (the MIG guide's placement table). Find a mix your planner accepts that the hardware can't place, and say how you'd detect it before rollout (hint: try it with `nvidia-smi mig -cgi` on the box, or encode the placement table).
2. Generate a mig-parted config for your mix with `python -m partitioning.cli mig …` and compare it with `platform/partitioning/k8s/mig-config.yaml`.
3. **Beat first-fit decreasing.** `test_optimal_two_gpus` (xfail) has a mix that fits on 2 GPUs, but FFD by memory uses 3: it pairs the two `3g.40gb` profiles and strands a compute slice. Make your planner find the 2-GPU packing, either with an exact search (the instances are tiny: branch and bound over GPUs) or with a smarter ordering that balances compute and memory. Then remove the xfail.
