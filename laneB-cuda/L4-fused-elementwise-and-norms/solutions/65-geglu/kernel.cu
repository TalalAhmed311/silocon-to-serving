// LeetGPU #65 Gaussian Error Gated Linear Unit — Lane B L4 solution. input [a | b] (2·n floats);
// output[i] = gelu(a[i]) · b[i], exact GELU: gelu(x) = x · Φ(x) = 0.5·x·(1 + erf(x/√2)). Check the statement for
// which half is gated and whether the tanh approximation is expected (the test tolerance would differ: README).
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

__global__ void geglu_k(const float* __restrict__ in, float* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = gelu_erf(in[i]) * in[n + i];
}

void solve(const float* input, float* output, int n) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  geglu_k<<<sms * 8, 256>>>(input, output, n);
}
