# Exercise 3: FlashAttention-2 forward + causal (T2, L6 exit check)

Swap the loops. **One block per tile of query rows**, inner loop over K/V tiles. Running `(m, l, o)` stays in **registers** for the whole loop. Rescale `o` and `l` **once per K/V tile** (compute the tile's scores first, then its max), not once per key. With the causal mask, stop the K/V loop at the end of the query tile, because later tiles are fully masked.

The tests are the same as exercises 1–2 (full and causal, awkward N). `--bench` prints ms and TFLOP/s at N = 4096. **L6 exit check:** run `examples/attention_bench.cu`, whose FA-2 vs naive column must show ≥ 5× at N = 4096. `TODO(run-on: g4dn.xlarge)`

**Stretch:** move the two matmuls of each tile (`QKᵀ` and `PV`) onto tensor cores with `mma.sync` (P5.7). The softmax rescale then has to work on the accumulator fragment layout, and that's where FA-2's "fewer non-matmul FLOPs" discussion becomes concrete.
