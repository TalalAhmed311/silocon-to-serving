// LeetGPU #18 Sparse Matrix-Vector Multiplication — Lane B L3 solution, CSR format: y = A x.
// row_ptr[rows+1], col_idx[nnz], vals[nnz]. If your statement passes a DENSE matrix with a sparsity hint, build CSR
// first (count nonzeros per row → scan (#16) → fill: compaction (#72) per row) — and measure whether it pays off.
#include <cuda_runtime.h>

// One warp per row: lanes stride over the row's nonzeros (coalesced vals/col_idx reads), then a shuffle reduction.
__global__ void spmv_warp_per_row(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const float* __restrict__ vals,
                                  const float* __restrict__ x, float* __restrict__ y, int rows) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;                       // whole warp exits together: shuffles below stay convergent
  const int s = row_ptr[warp], e = row_ptr[warp + 1];
  float acc = 0.f;
  for (int k = s + lane; k < e; k += 32) acc = fmaf(vals[k], __ldg(&x[col_idx[k]]), acc);   // x gathers: read-only path
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[warp] = acc;
}

// One thread per row: better when rows are very short (avg nnz/row < ~4), where a warp would be mostly idle.
__global__ void spmv_thread_per_row(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const float* __restrict__ vals,
                                    const float* __restrict__ x, float* __restrict__ y, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float acc = 0.f;
  for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) acc = fmaf(vals[k], x[col_idx[k]], acc);
  y[r] = acc;
}

void solve(const int* row_ptr, const int* col_idx, const float* vals, const float* x, float* y, int rows, int nnz) {
  const double avg = rows ? double(nnz) / rows : 0;
  if (avg >= 4) spmv_warp_per_row<<<(rows + 7) / 8, 256>>>(row_ptr, col_idx, vals, x, y, rows);   // 8 warps (rows) per block
  else spmv_thread_per_row<<<(rows + 255) / 256, 256>>>(row_ptr, col_idx, vals, x, y, rows);
}
