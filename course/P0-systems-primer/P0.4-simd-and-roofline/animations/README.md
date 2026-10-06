# Animations used in P0.4

- [SIMD lanes](../../../../animations/p0-simd-lanes.html) (`animations/p0-simd-lanes.html`): a scalar loop vs an 8-wide AVX2 add, then an aligned load vs one that splits two cache lines.
- [Roofline](../../../../animations/roofline.html) (`animations/roofline.html`): drag the arithmetic intensity, or pick a kernel, and watch it move from memory-bound to compute-bound. The machine presets include "your laptop", where you enter the numbers from `02_fma_peak` and `03_stream`.
