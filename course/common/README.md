# course/common — shared helpers

Small helpers used by every module, so each example stays focused on its one idea.

| File | Language | What |
|---|---|---|
| `include/s2s/check.hpp` | C++17 | `CHECK(cond)`, `CHECK_NEAR(a, b, rtol, atol)`, `check_allclose(...)`, test registry + `main` |
| `include/s2s/bench.hpp` | C++17 | warm-up + repeat timing, median/p90, Markdown table printer, JSON writer |
| `include/s2s/aligned.hpp` | C++17 | `aligned_buffer<T>`: 64-byte aligned, move-only owner |
| `python/s2s/bench.py` | Python | the same table/JSON contract for Python benches |
| `python/s2s/tables.py` | Python | Markdown table formatting |

**Bench contract.** Every bench prints a Markdown table whose columns include `problem size | time | GB/s or GFLOP/s | % of peak`. It also writes `results/<name>.json` with this shape:

```json
{"bench": "transpose", "hardware": "...", "unit": "GB/s",
 "rows": [{"size": 4096, "median_ms": 1.2, "p90_ms": 1.3, "rate": 12.5, "pct_peak": 61.0}]}
```

The site's charts read this file. "Peak" always comes from a measurement made in the same run, such as a STREAM-style copy or an FMA loop, or from a cited spec. It is never typed in by hand.

To use these from CMake:

```cmake
target_include_directories(my_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)
```
