// 02_occupancy.cu — how registers, shared memory and block size limit resident warps per SM (sm_75+).
// Expected: a table of block size | max active blocks/SM | occupancy % for three kernels that differ only in register
// and shared-memory use, plus the API's suggested block size. Compare with `nvcc --resource-usage`.
#include <cstdio>

#include "s2s_cuda.cuh"

__global__ void light(float* x) { x[blockIdx.x * blockDim.x + threadIdx.x] *= 2.f; }

__global__ void heavy_regs(float* x) {           // many live values → many registers per thread
  float r[48];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int k = 0; k < 48; ++k) r[k] = x[i] * k;
  float s = 0;
#pragma unroll
  for (int k = 0; k < 48; ++k) s += r[k] * r[(k * 7) % 48];
  x[i] = s;
}

__global__ void heavy_smem(float* x) {           // 32 KB of static shared memory per block
  __shared__ float buf[8192];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[threadIdx.x] = x[i];
  __syncthreads();
  x[i] = buf[(threadIdx.x + 1) % blockDim.x];
}

template <class K>
void report(const char* name, K kernel) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  std::printf("\n%s: %d regs/thread, %zu B static smem\n| block | blocks/SM | warps/SM | occupancy |\n|---|---|---|---|\n",
              name, a.numRegs, a.sharedSizeBytes);
  for (int bs : {64, 128, 256, 512, 1024}) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, bs, 0));
    const int warps = blocks * bs / 32, maxw = p.maxThreadsPerMultiProcessor / 32;
    std::printf("| %d | %d | %d | %.0f%% |\n", bs, blocks, warps, 100.0 * warps / maxw);
  }
  int min_grid = 0, best = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best, kernel, 0, 0));
  std::printf("API suggestion: block %d (min grid for full occupancy: %d blocks)\n", best, min_grid);
}

int main() {
  std::printf("GPU: %s\n", s2s::gpu_name().c_str());
  report("light", light);
  report("heavy_regs", heavy_regs);
  report("heavy_smem", heavy_smem);
  std::printf("\nHigher occupancy hides latency only up to a point: measure, don't maximise blindly (P5.3).\n");
}
