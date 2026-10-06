# Exercise 3 — The `pct_of_peak` helper (easy)

Every bench from P2 on prints a "% of peak" column. Implement it once, correctly, in `peak.py`:

```python
pct_of_peak(measured, gpu_name, kind, precision="fp16") -> float | None
#   kind="bandwidth": measured in GB/s vs hbm_gbs
#   kind="compute":   measured in TFLOP/s vs <precision>_dense_tflops
#   returns None when the spec value is null (never invent a peak)
annotate(rate, gpu_name, kind, precision="fp16") -> str
#   "123.4 (41% of UNVERIFIED peak)" or "123.4 (41% of peak)" once that GPU is VERIFIED, or "123.4 (no spec)"
```

**Test:** unit tests with the real YAML, including a `null` (A10G) and the UNVERIFIED annotation.
