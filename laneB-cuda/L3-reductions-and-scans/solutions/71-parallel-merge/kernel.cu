// LeetGPU #71 Parallel Merge — Lane B L3 solution. C = merge of sorted A (m) and sorted B (n), ascending; ties take A first.
// Merge path: thread t produces outputs [t*ITEMS, (t+1)*ITEMS); a binary search on its diagonal finds where to start.
#include <cuda_runtime.h>

constexpr int ITEMS = 8;

// Number of elements taken from A among the first `diag` outputs of the merge.
__device__ int merge_path(const float* A, int m, const float* B, int n, int diag) {
  int lo = diag > n ? diag - n : 0, hi = diag < m ? diag : m;
  while (lo < hi) {                       // find the smallest i with A[i] > B[diag - i - 1]
    const int i = (lo + hi) >> 1;
    if (A[i] <= B[diag - i - 1]) lo = i + 1;   // `<=` sends ties to A first (stable)
    else hi = i;
  }
  return lo;
}

__global__ void merge_kernel(const float* __restrict__ A, int m, const float* __restrict__ B, int n, float* __restrict__ C) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const int total = m + n;
  const long long d0 = t * ITEMS;
  if (d0 >= total) return;
  const int d1 = int(d0 + ITEMS < total ? d0 + ITEMS : total);
  int i = merge_path(A, m, B, n, int(d0));
  int j = int(d0) - i;
  for (int d = int(d0); d < d1; ++d) {   // sequential merge of my ITEMS outputs
    if (j >= n || (i < m && A[i] <= B[j])) C[d] = A[i++];
    else C[d] = B[j++];
  }
}

void solve(const float* A, const float* B, float* C, int M, int N) {
  const long long threads = (long long(M) + N + ITEMS - 1) / ITEMS;
  if (threads == 0) return;
  merge_kernel<<<int((threads + 255) / 256), 256>>>(A, M, B, N, C);
}
