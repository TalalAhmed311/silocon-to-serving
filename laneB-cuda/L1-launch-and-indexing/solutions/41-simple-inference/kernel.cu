// LeetGPU #41 Simple Inference — Lane B L1 solution (harness convention: x[batch,in], W[out,in], b[out], y[batch,out]).
#include <cuda_runtime.h>

__global__ void linear(const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ b,
                       float* __restrict__ y, int batch, int in, int out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= batch || o >= out) return;
  float s = b[o];
  for (int i = 0; i < in; ++i) s += x[r * in + i] * W[o * in + i];
  y[r * out + o] = s;
}

void solve(const float* x, const float* W, const float* b, float* y, int batch, int in, int out) {
  dim3 block(32, 8), grid((out + 31) / 32, (batch + 7) / 8);
  linear<<<grid, block>>>(x, W, b, y, batch, in, out);
}
