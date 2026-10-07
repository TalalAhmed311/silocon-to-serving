// LeetGPU #55 Attention with Linear Biases — Lane B L6 solution. Q, K, V: [H, N, d], causal; score(i, j) +=
// −m_h·(i − j) with the ALiBi slopes m_h = 2^(−8(h+1)/H) (for H a power of two; Press et al.). Slopes computed on the
// host and passed in. Check the statement for the slope formula and whether it is causal.
#include "../_shared/attn_core.cuh"

#include <cmath>
#include <vector>

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d, int H) {
  std::vector<float> slopes(H);
  for (int h = 0; h < H; ++h) slopes[h] = std::pow(2.f, -8.f * (h + 1) / H);
  float* d_slopes = nullptr;
  cudaMalloc(&d_slopes, sizeof(float) * H);
  cudaMemcpy(d_slopes, slopes.data(), sizeof(float) * H, cudaMemcpyHostToDevice);
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.H = p.Hkv = H; p.Lq = p.Lk = N; p.d = d; p.causal = true; p.alibi = d_slopes;
  attn::set_bhld(p);
  attn::launch(p);
  cudaFree(d_slopes);                                  // cudaFree synchronises: safe after the launch
}
