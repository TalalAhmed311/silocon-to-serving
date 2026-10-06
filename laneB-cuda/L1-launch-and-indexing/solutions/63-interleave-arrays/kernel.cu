// LeetGPU #63 Interleave Arrays — Lane B L1 solution. A, B: N floats; out: 2N floats. Device pointers.
#include <cuda_runtime.h>

__global__ void interleave(const float* __restrict__ A, const float* __restrict__ B, float2* __restrict__ out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = make_float2(A[i], B[i]);  // one 8-byte store == out[2i], out[2i+1]
}

void solve(const float* A, const float* B, float* output, int N) {
  interleave<<<(N + 255) / 256, 256>>>(A, B, reinterpret_cast<float2*>(output), N);  // cudaMalloc'd: 8-byte aligned
}
