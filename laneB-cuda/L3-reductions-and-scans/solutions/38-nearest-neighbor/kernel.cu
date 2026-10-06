// LeetGPU #38 Nearest Neighbor — Lane B L3 solution. points: N x 3 floats (x, y, z interleaved);
// indices[i] = argmin_{j != i} |p_i - p_j|^2 (ties → smaller j). Brute force O(N^2), tiled through shared memory.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int TILE = 256;

__global__ void nn_kernel(const float* __restrict__ pts, int* __restrict__ idx, int N) {
  __shared__ float sx[TILE], sy[TILE], sz[TILE];      // structure-of-arrays in smem: no bank conflicts on the reads
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float px = 0, py = 0, pz = 0;
  if (i < N) { px = pts[3 * i]; py = pts[3 * i + 1]; pz = pts[3 * i + 2]; }
  float best = FLT_MAX;
  int best_j = -1;
  for (int base = 0; base < N; base += TILE) {
    const int j = base + threadIdx.x;                 // every thread loads one point of the tile (coalesced-ish AoS)
    if (j < N) { sx[threadIdx.x] = pts[3 * j]; sy[threadIdx.x] = pts[3 * j + 1]; sz[threadIdx.x] = pts[3 * j + 2]; }
    __syncthreads();
    const int lim = min(TILE, N - base);
    for (int k = 0; k < lim; ++k) {                   // all threads read the same sx[k]: a broadcast
      const float dx = sx[k] - px, dy = sy[k] - py, dz = sz[k] - pz;
      const float d = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
      const int jj = base + k;
      if (jj != i && d < best) { best = d; best_j = jj; }   // strict < with increasing jj → smallest index on ties
    }
    __syncthreads();
  }
  if (i < N) idx[i] = best_j;
}

void solve(const float* points, int* indices, int N) {
  nn_kernel<<<(N + TILE - 1) / TILE, TILE>>>(points, indices, N);
}
