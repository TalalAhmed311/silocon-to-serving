#pragma once
#include <d4/reduce.cuh>
inline void reduce_sum(int rung, const float* in, float* out, float* partial, long long n) {
  d4::reduce_sum(rung, in, out, partial, n);
}
