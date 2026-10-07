#pragma once
#include <d4/elementwise.cuh>
inline void vadd(const float* a, const float* b, float* c, long long n) { d4::vadd(a, b, c, n, /*vec4=*/false); }
