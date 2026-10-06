// LeetGPU #75 Sparse Matrix-Dense Matrix Multiplication — Lane B L3 solution: C[M x P] = A_csr[M x N] * B[N x P],
// row-major B and C. Our signature (CSR input); check the statement.
#include <cuda_runtime.h>

// Block per row of A (grid-stride over rows); threads cover C's columns. For each nonzero a(r,k), every thread reads
// B[k, col] — a contiguous row of B across the block → coalesced. a(r,k) and k are broadcast via shared memory.
constexpr int THREADS = 128, CHUNK = 128;

__global__ void spmm_row(const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const float* __restrict__ vals,
                         const float* __restrict__ B, float* __restrict__ C, int M, int P) {
  __shared__ int sk[CHUNK];
  __shared__ float sv[CHUNK];
  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    const int s = row_ptr[r], e = row_ptr[r + 1];
    for (int c0 = 0; c0 < P; c0 += THREADS) {          // column strip handled by this block
      const int col = c0 + threadIdx.x;
      float acc = 0.f;
      for (int k0 = s; k0 < e; k0 += CHUNK) {          // stage the row's nonzeros through shared memory
        const int cnt = min(CHUNK, e - k0);
        __syncthreads();
        if (threadIdx.x < cnt) { sk[threadIdx.x] = col_idx[k0 + threadIdx.x]; sv[threadIdx.x] = vals[k0 + threadIdx.x]; }
        __syncthreads();
        if (col < P)
          for (int q = 0; q < cnt; ++q) acc = fmaf(sv[q], B[size_t(sk[q]) * P + col], acc);
      }
      if (col < P) C[size_t(r) * P + col] = acc;
    }
  }
}

void solve(const int* row_ptr, const int* col_idx, const float* vals, const float* B, float* C, int M, int N, int P) {
  (void)N;
  const int blocks = M < 65535 ? M : 65535;
  if (blocks > 0) spmm_row<<<blocks, THREADS>>>(row_ptr, col_idx, vals, B, C, M, P);
}
