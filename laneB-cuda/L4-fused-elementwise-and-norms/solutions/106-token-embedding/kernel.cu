// LeetGPU #106 Token Embedding Layer — Lane B L4 solution. out[t, :] = table[ids[t], :], table: [V, D] fp32.
#include <cuda_runtime.h>

// One block per token; float4 copies of the row when D % 4 == 0. Pure memory movement: the gather is by row, so each
// row read is contiguous and coalesced — the only irregularity is WHICH row.
__global__ void embed_k(const int* __restrict__ ids, const float* __restrict__ table, float* __restrict__ out, int D, int V) {
  const int t = blockIdx.x;
  const int id = ids[t];
  float* dst = out + (size_t)t * D;
  if (id < 0 || id >= V) {                                   // out-of-range id: zeros (or assert, per the statement)
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = 0.f;
    return;
  }
  const float* src = table + (size_t)id * D;
  if ((D & 3) == 0) {
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(dst);
    for (int d = threadIdx.x; d < D / 4; d += blockDim.x) d4[d] = s4[d];
  } else {
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
  }
}

void solve(const int* token_ids, const float* table, float* output, int T, int D, int V) {
  embed_k<<<T, 128>>>(token_ids, table, output, D, V);
}
