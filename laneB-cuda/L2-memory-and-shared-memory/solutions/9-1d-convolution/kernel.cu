// LeetGPU #9 1D Convolution — Lane B L2 solution. Valid correlation: out[i] = sum_j in[i+j]*k[j], i < N-K+1.
#include <cuda_runtime.h>

constexpr int BLOCK = 256, MAX_K = 2048;
__constant__ float c_kernel[MAX_K];  // broadcast to all threads of a warp reading the same tap

__global__ void conv1d_smem(const float* __restrict__ in, float* __restrict__ out, int N, int K) {
  extern __shared__ float tile[];    // BLOCK + K - 1 floats: this block's inputs plus the right halo
  const int base = blockIdx.x * BLOCK, out_n = N - K + 1;
  for (int t = threadIdx.x; t < BLOCK + K - 1; t += BLOCK)
    tile[t] = (base + t < N) ? in[base + t] : 0.0f;
  __syncthreads();
  const int i = base + threadIdx.x;
  if (i >= out_n) return;
  float s = 0.0f;
  for (int j = 0; j < K; ++j) s += tile[threadIdx.x + j] * c_kernel[j];
  out[i] = s;
}

__global__ void conv1d_global(const float* __restrict__ in, const float* __restrict__ k, float* __restrict__ out, int N, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N - K + 1) return;
  float s = 0.0f;
  for (int j = 0; j < K; ++j) s += in[i + j] * k[j];
  out[i] = s;
}

// input: N floats, kernel: K floats, output: N-K+1 floats (device pointers).
void solve(const float* input, const float* kernel, float* output, int N, int K) {
  const int out_n = N - K + 1, blocks = (out_n + BLOCK - 1) / BLOCK;
  if (out_n <= 0) return;
  if (K <= MAX_K) {
    cudaMemcpyToSymbol(c_kernel, kernel, K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    conv1d_smem<<<blocks, BLOCK, (BLOCK + K - 1) * sizeof(float)>>>(input, output, N, K);
  } else {
    conv1d_global<<<blocks, BLOCK>>>(input, kernel, output, N, K);
  }
}
