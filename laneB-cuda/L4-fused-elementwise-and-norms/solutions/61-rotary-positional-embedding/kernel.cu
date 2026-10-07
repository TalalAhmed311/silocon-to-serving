// LeetGPU #61 Rotary Positional Embedding — Lane B L4 solution. x: [T, H, D] fp32 (T tokens, H heads, D even);
// pos: [T] int positions. "rotate_half" convention (Llama / HF): pairs (i, i + D/2) rotate by angle
//   θ_i(p) = p · base^(−2i/D),  x'_i = x_i·cos − x_{i+D/2}·sin,  x'_{i+D/2} = x_{i+D/2}·cos + x_i·sin.
// The other common convention rotates adjacent pairs (2i, 2i+1) — check the statement; they are NOT interchangeable.
#include <cuda_runtime.h>
#include <cmath>

__global__ void rope_k(float* __restrict__ x, const int* __restrict__ pos, int H, int D, float base) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i in [0, D/2)
  if (i >= D / 2) return;
  const float inv_freq = powf(base, -2.f * i / D);             // on the fly; a precomputed cos/sin table also works
  float s, c;
  sincosf(pos[t] * inv_freq, &s, &c);
  float* v = x + ((size_t)t * H + h) * D;
  const float a = v[i], b = v[i + D / 2];
  v[i] = a * c - b * s;
  v[i + D / 2] = b * c + a * s;
}

void solve(float* x, const int* positions, int T, int H, int D, float base) {
  rope_k<<<dim3(T, H), D / 2>>>(x, positions, H, D, base);
}
