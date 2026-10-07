#pragma once
#include <d4/softmax.cuh>
// v3 (row in registers: one read) for rows ≤ 1024, else v2 (online max+sum pass, then normalize pass).
inline void softmax_rows(const float* x, float* y, int rows, int cols) { d4::softmax(3, x, y, rows, cols); }
