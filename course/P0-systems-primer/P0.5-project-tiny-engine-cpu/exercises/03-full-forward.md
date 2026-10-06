# Exercise 3 — Full forward pass (medium)

With exercises 1 and 2 done, the whole engine runs on your ops. This exercise is about **verifying** a model end to end, and about reading `engine.hpp` until you could rewrite it from memory.

1. `ctest -R 03` runs `engine::Engine` (built with your `ops.hpp`) on the fixture's tiny model:
   - **teacher-forced logits:** 12 fixed input ids, every step's logits vs NumPy, `rtol=atol=1e-4`
   - **greedy tokens:** 48 greedy steps vs NumPy. A divergence is accepted only at a reference near-tie (< 1e-4 gap).
2. Then run your binary: `build/p05-ex/s2s-engine-ex --model build/p05-fixtures/tiny --prompt-ids "1 2 3 4" --steps 32`.
3. **Break it on purpose** to learn what the tests catch:
   - swap RoPE to the interleaved convention
   - use `h % n_kv_heads` for GQA
   - attend to `0..pos-1` instead of `0..pos`

   Note which test fails first, and by how much.

**Why 1e-4 and not 1e-6:** your matvec sums with 4 SIMD accumulators across threads, while NumPy's BLAS sums in its own order. Over 2 layers, fp32 rounding differences of ~1e-7 relative grow to ~1e-5–1e-4 on logits of magnitude ~10.
