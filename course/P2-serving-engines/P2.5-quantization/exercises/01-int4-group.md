# Exercise 1 — INT4 group quantization in NumPy (T0)

Implement in [`int4.py`](int4.py):

```python
quantize(w, group=128, symmetric=True) -> (q, scale, zero, shape)
    # w: [rows, cols], cols % group == 0. Per row, per group of `group` columns:
    #   symmetric:  scale = max|w| / 7,               q = clip(round(w / scale), -8, 7),  zero = 0
    #   asymmetric: scale = (max − min) / 15,         zero = round(−min / scale),
    #               q = clip(round(w / scale) + zero, 0, 15)
    # q is stored as int8 (one value per element; packing two nibbles per byte is optional, exercise 1b)
dequantize(q, scale, zero, shape) -> w_hat
```

**Test (`test_int4.py`):**
- the elementwise error is ≤ scale/2 + 1e-7
- asymmetric is no worse than symmetric on a skewed (all-positive) group
- smaller groups give lower error on data with outliers
- round trip of an all-zero group
- **1b (optional):** `pack(q)` / `unpack(packed)` two-nibbles-per-byte round trip
