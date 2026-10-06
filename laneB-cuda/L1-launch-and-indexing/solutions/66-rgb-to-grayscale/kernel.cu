// LeetGPU #66 RGB to Grayscale — Lane B L1 solution. input: 3*width*height floats (RGB interleaved); output: width*height.
#include <cuda_runtime.h>

// Stand-in BT.601 weights — replace with the statement's coefficients if they differ.
constexpr float kR = 0.299f, kG = 0.587f, kB = 0.114f;

__global__ void rgb2gray(const float* __restrict__ in, float* __restrict__ out, int n) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n) out[p] = kR * in[3 * p] + kG * in[3 * p + 1] + kB * in[3 * p + 2];
}

void solve(const float* input, float* output, int width, int height) {
  const int n = width * height;
  rgb2gray<<<(n + 255) / 256, 256>>>(input, output, n);
}
