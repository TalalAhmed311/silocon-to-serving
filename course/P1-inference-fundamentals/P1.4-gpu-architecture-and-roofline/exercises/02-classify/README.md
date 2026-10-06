# Exercise 2 — Ridge points and bound classification (easy)

Implement in `classify.py`:

- `ridge(peak_tflops, gbs) -> float` in FLOP/byte
- `attainable_tflops(intensity, peak_tflops, gbs) -> float`
- `bound(intensity, peak_tflops, gbs) -> "memory" | "compute"`
- `decode_intensity(batch, bytes_per_param=2) -> float`, with weights dominating: ≈ `2·batch / bytes_per_param`
- `prefill_intensity(tokens, bytes_per_param=2) -> float`, ≈ `2·tokens / bytes_per_param`

**Test:** fixture GPUs (round numbers, independent of the YAML). Also checks that FP8 doubles decode intensity, and that at batch 1 every GPU in `gpu_specs.yaml` is memory-bound for decode.
