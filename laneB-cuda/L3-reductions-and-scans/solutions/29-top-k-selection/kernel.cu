// LeetGPU #29 Top-K Selection — Lane B L3 solution. output[0..k) = the k largest of input[0..N), descending.
// Rounds of "sort each 2048-tile in shared memory, keep its top K'" (K' = next pow2 >= k) shrink the candidate set by
// 2048/K' per round; the last round is one tile. Our signature; check the statement (order of the output, ties).
#include <cuda_runtime.h>
#include <cfloat>

constexpr int THREADS = 512, TILE = 2048;

__device__ __forceinline__ void stage_desc(float* s, unsigned j, unsigned k) {
  for (unsigned t = threadIdx.x; t < TILE / 2; t += blockDim.x) {
    const unsigned i = 2 * t - (t & (j - 1)), l = i + j;
    const bool down = (i & k) == 0;              // LOCAL index: every tile ends up sorted the same way (descending)
    const float a = s[i], b = s[l];
    if ((a < b) == down) { s[i] = b; s[l] = a; }
  }
}

// Sort each tile of `in` (n valid elements, rest = -inf) descending and write its first `keep` to out[tile*keep...].
__global__ void tile_topk(const float* __restrict__ in, int n, float* __restrict__ out, int keep) {
  __shared__ float s[TILE];
  const long long base = (long long)blockIdx.x * TILE;
  for (int t = threadIdx.x; t < TILE; t += blockDim.x) s[t] = base + t < n ? in[base + t] : -FLT_MAX;
  __syncthreads();
  for (unsigned k = 2; k <= TILE; k <<= 1)
    for (unsigned j = k >> 1; j > 0; j >>= 1) { stage_desc(s, j, k); __syncthreads(); }
  for (int t = threadIdx.x; t < keep; t += blockDim.x) out[(long long)blockIdx.x * keep + t] = s[t];
}

void solve(const float* input, float* output, int N, int k) {
  int keep = 1;
  while (keep < k) keep <<= 1;
  if (keep > TILE / 2) {   // k > 1024: this scheme can't shrink the set; use a full sort instead (see README)
    return;                // TODO(learner): fall back to #15's global bitonic sort (descending) and copy k
  }
  const float* cur = input;
  int n = N;
  float *bufA = nullptr, *bufB = nullptr;
  const int max_tiles = (N + TILE - 1) / TILE;
  cudaMalloc(&bufA, size_t(max_tiles) * keep * sizeof(float));
  cudaMalloc(&bufB, size_t(max_tiles) * keep * sizeof(float));
  float* dst = bufA;
  while (true) {
    const int tiles = (n + TILE - 1) / TILE;
    if (tiles == 1) {   // final round: one tile holds every candidate; its sorted prefix is the answer
      tile_topk<<<1, THREADS>>>(cur, n, dst, keep);
      cudaMemcpy(output, dst, size_t(k) * sizeof(float), cudaMemcpyDeviceToDevice);
      break;
    }
    tile_topk<<<tiles, THREADS>>>(cur, n, dst, keep);
    cur = dst;
    n = tiles * keep;
    dst = (dst == bufA) ? bufB : bufA;
  }
  cudaFree(bufA);
  cudaFree(bufB);
}
