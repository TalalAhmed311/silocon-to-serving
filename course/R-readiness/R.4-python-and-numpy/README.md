# R.4 — Python and NumPy for this course

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Linux or macOS, no GPU |
| **Time** | ≈30 min reading + ≈3 h hands-on |
| **Prerequisites** | You can write a loop and a function in some language. R.3 (running `uv run pytest`) helps. |
| **You will build** | 4 tested NumPy exercises: shape and byte arithmetic, views, broadcasting, vectorising a loop |

## Learning objectives

By the end of this module you can:

1. Write the Python the course uses: functions, dicts, comprehensions, dataclasses and type hints.
2. Describe an `ndarray` as a typed flat buffer plus a shape plus strides, and compute its size in bytes from shape and dtype.
3. Tell a view from a copy, and say which operations are free (slice, transpose, reshape) and which allocate.
4. Predict the result shape of a broadcast and write loop-free array code.
5. Estimate the FLOPs of a matrix multiply, and write tests that compare floats with tolerances.

## Why this matters for inference engineering

The course writes its engines in C++ and CUDA, and uses Python for **reference implementations** (a slow, obviously correct version) and **tests** (compare the fast version against the reference). `platform/engine/v0/reference/llama_numpy.py` is a whole Llama forward pass in NumPy. If you can read NumPy shapes, you can read that file, and every model-size number in the course is shape-times-dtype arithmetic.

---

## 1. Why Python next to C++

A fast kernel is easy to get wrong in ways that still run. So each engine piece has a short NumPy twin that you trust, and a test that runs both on the same inputs and checks they agree. `pytest` runs those tests. R.3 showed how to run them with `uv run pytest`.

## 2. Python essentials

```python
from dataclasses import dataclass

@dataclass
class Config:                       # a class that is mostly named fields
    layers: int = 32
    kv_heads: int = 8

def kv_per_token(c: Config, head_dim: int = 128, bytes_per: int = 2) -> int:   # type hints document intent
    return 2 * c.layers * c.kv_heads * head_dim * bytes_per

sizes = {name: len(name) for name in ["q", "k", "v"]}     # dict comprehension
squares = [i * i for i in range(5)]                        # list comprehension: [0, 1, 4, 9, 16]
```

Type hints are not enforced when the code runs. They tell the reader (and tools) what is expected.

## 3. An `ndarray` is a buffer plus a shape

```python
import numpy as np
a = np.arange(6, dtype=np.float32).reshape(2, 3)
a.shape, a.dtype, a.strides      # ((2, 3), dtype('float32'), (12, 4))
```

NumPy stores the six numbers in one flat run of memory and remembers a **shape** (2, 3) and **strides** (12, 4): to move one row, jump 12 bytes; one column, 4 bytes. This is exactly P0.1's `offset = i * stride0 + j * stride1`.

## 4. The dtype decides the bytes

`bytes = number of elements x bytes per element`, and `a.nbytes` reports it.

| dtype | bytes per element |
|---|---|
| float32 | 4 |
| float16, bfloat16 | 2 |
| int8 | 1 |

A real example: one Llama-3-8B layer's `q_proj` weight is 4096 x 4096 = 16,777,216 elements. In float32 that is 67,108,864 bytes = **64 MiB**. In bfloat16 it is **32 MiB**. (NumPy has no bfloat16 type; the course's tools handle it by byte count.)

## 5. Indexing, slicing, views and copies

`a[1]`, `a[:, 0]` and `a[::2]` return **views**: new shape and strides over the same buffer. Writing through a view changes the original. `a.copy()` makes a new buffer. Fancy indexing (`a[[0, 2]]`, boolean masks) always copies. Check with `np.shares_memory(a, b)`.

## 6. Reshape and transpose change strides, not data

`a.T` swaps the strides. `a.reshape(...)` gives a view when the memory layout allows it, otherwise it copies. After `a.T` the data is no longer in the "natural" order, so walking it row by row jumps in memory. `np.ascontiguousarray(a.T)` pays for a real copy once.

## 7. Broadcasting

Operations on arrays of different shapes work when the shapes line up **from the right**. For each pair of dimensions they must be equal, or one must be 1 (that one is stretched). A missing dimension counts as 1.

```
x (4, 3)      x - mean   where mean has shape (3,)   -> (4, 3)   ok
x (4, 3)      x - mean   where mean has shape (4,)   -> error: 3 vs 4
x (4, 3)      x - mean   where mean has shape (4, 1) -> (4, 3)   ok
```

Nothing is copied: NumPy uses a stride of 0 for the stretched dimension.

## 8. Vectorise

A Python loop pays interpreter overhead on every element. One array expression runs a compiled loop. Replace `for` over elements with whole-array operations, then **measure** with `time.perf_counter()` on your machine. The size of the speed-up depends on array size and hardware, so this course never quotes one.

## 9. Matrix multiply

`A @ B` with shapes `(m, k) @ (k, n)` gives `(m, n)`. Each of the `m x n` outputs is a dot product of length `k`: `k` multiplies and about `k` adds, counted as `2k` FLOPs. So the cost is

```
FLOPs = 2 * m * k * n
```

One token through `q_proj` is `(1, 4096) @ (4096, 4096)`: 2 x 1 x 4096 x 4096 = 33,554,432 FLOPs.

## 10. pytest and floating point

A test is a function named `test_*` that uses `assert`. Floats from different but valid computations differ in the last bits, so do not compare with `==`:

```python
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
```

passes when `|actual - expected| <= atol + rtol * |expected|`. Use `rtol` for relative error and add a small `atol` when the expected value can be 0.

---

## Exercises

```bash
uv run pytest course/R-readiness/R.4-python-and-numpy
S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.4-python-and-numpy
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Shapes and bytes](exercises/01-shapes-and-bytes/README.md) | easy | hand-computed sizes and FLOPs |
| 2 | [Views, not copies](exercises/02-views/README.md) | easy | `np.shares_memory`, strides |
| 3 | [Broadcasting](exercises/03-broadcast/README.md) | medium | against an explicit per-row formula |
| 4 | [Vectorise a loop](exercises/04-vectorize/README.md) | medium | matches the loop within `rtol=1e-6` |

## Common mistakes

- **Expecting `b = a[:, 0]` to be a copy.** Editing `b` edits `a`. Use `.copy()`.
- **Mean over the wrong axis.** `axis=1` collapses columns. Use `keepdims=True` to keep the result broadcastable.
- **Comparing floats with `==`.** Use `assert_allclose`.
- **Integer overflow in small dtypes.** `int8` wraps at 127 to -128. Check before you cast.
- **Trusting a speed-up you did not measure.**

## Go further

- The Python tutorial: https://docs.python.org/3/tutorial/
- NumPy, the absolute basics for beginners: https://numpy.org/doc/stable/user/absolute_beginners.html

Both links are UNVERIFIED from the build environment (no outbound access).

## Next

[R.5 — The maths and ML you need](../R.5-maths-and-ml-basics/README.md)
