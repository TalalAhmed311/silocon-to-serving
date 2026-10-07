// Exercise 3 starter: inclusive scan of n floats. Step 1: a block scan (one 2048-element tile per block: 8 items per
// thread serially, then a scan of the 256 thread totals). Step 2: scan-then-propagate for any n (tile totals →
// recursive scan → add carries). Optional step 3: single-pass decoupled look-back (see d4/scan.cuh only afterwards).
#pragma once
#include <cuda_runtime.h>

inline void inclusive_scan(const float* in, float* out, int n) {
  (void)in; (void)out; (void)n;   // TODO
}
