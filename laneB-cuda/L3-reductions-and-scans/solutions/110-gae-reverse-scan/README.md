# 110: Parallel Reverse Scan, GAE (L3, stretch)

Problem: [LeetGPU #110](../../leetgpu-map.md). Generalized Advantage Estimation runs **backwards** in time: `A[t] = δ[t] + γλ(1 − done[t])·A[t+1]`.

**Hint ladder**

1. Compute `δ[t]` first. It's elementwise, so fuse it into the packing kernel.
2. `A[t] = c_t·A[t+1] + δ[t]` is #82's affine recurrence, run in reverse. **Reverse the index** while packing (`j = N−1−t`), scan forward, and un-reverse while unpacking. No new scan code is needed.
3. Episode boundaries need no special handling: `done[t] = 1` makes `c_t = 0`, and the affine composition then forgets everything after t. The test's `γ = λ = 1` case checks exactly that.

**Solution outline:** `pack_reversed` (δ and `c = γλ(1−done)`, reversed) → generic `inclusive_scan` with `Compose` → `unpack_reversed`.

**Why this is fast:** one fused map pass, then a memory-bound scan of 8-byte pairs. For RL training this replaces an O(N) Python loop per batch.
