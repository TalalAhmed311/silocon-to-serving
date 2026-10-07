// LeetGPU #54 Swish-Gated Linear Unit — Lane B L4 solution. input: 2·n floats [a | b]; output[i] = silu(a[i]) · b[i],
// silu(x) = x · sigmoid(x). Our layout assumption (gate first half, up second half); check the statement.
#include <cuda_runtime.h>

__global__ void swiglu_k(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int n4 = n / 4;
  const float4* a4 = reinterpret_cast<const float4*>(in);
  const float4* b4 = reinterpret_cast<const float4*>(in + n);       // aligned only if n % 4 == 0 (checked by caller)
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const float4 a = a4[i], b = b4[i];
    float4 o;
    o.x = a.x / (1.f + __expf(-a.x)) * b.x;
    o.y = a.y / (1.f + __expf(-a.y)) * b.y;
    o.z = a.z / (1.f + __expf(-a.z)) * b.z;
    o.w = a.w / (1.f + __expf(-a.w)) * b.w;
    reinterpret_cast<float4*>(out)[i] = o;
  }
}

__global__ void swiglu_scalar(const float* __restrict__ in, float* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float a = in[i];
    out[i] = a / (1.f + __expf(-a)) * in[n + i];
  }
}

void solve(const float* input, float* output, int n) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (n % 4 == 0) swiglu_k<<<sms * 8, 256>>>(input, output, n);
  else swiglu_scalar<<<sms * 8, 256>>>(input, output, n);
}
