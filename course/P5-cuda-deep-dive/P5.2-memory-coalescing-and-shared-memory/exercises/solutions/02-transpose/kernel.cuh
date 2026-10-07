#pragma once
#include <d4/transpose.cuh>
inline void transpose(const float* in, float* out, int rows, int cols) { d4::transpose(in, out, rows, cols, 3); }
