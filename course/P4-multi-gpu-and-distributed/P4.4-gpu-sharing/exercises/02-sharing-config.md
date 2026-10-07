# Exercise 2: time-slicing and MPS configs (T0)

The GPU Operator's device plugin reads a sharing config from a ConfigMap. `platform/partitioning/sharing_config.py` builds and validates them, and `test_sharing_config.py` checks both the generated configs and the ones in `platform/partitioning/k8s/time-slicing.yaml`.

1. Read the validator. Why is `failRequestsGreaterThanOne` required? (A pod requesting `nvidia.com/gpu: 2` on a time-sliced node would otherwise get two slices of the **same** GPU.)
2. Add a rule: with MPS, each client should also get a memory limit. Find how the device plugin exposes that for MPS (docs, UNVERIFIED), add it to `mps()`, and test it.
3. Write down, for each mode, what one misbehaving pod can do to the others: OOM, a hang, an Xid error, or a hog of SM time.
