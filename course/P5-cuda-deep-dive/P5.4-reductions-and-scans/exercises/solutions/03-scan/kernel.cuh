#pragma once
#include <d4/scan.cuh>
inline void inclusive_scan(const float* in, float* out, int n) { d4::scan_3phase(in, out, n); }
