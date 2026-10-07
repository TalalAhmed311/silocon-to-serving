// 01_hello_indices.cu — which thread am I? Prints the index mapping for a tiny 2D launch, then the SM each block ran on.
// Build: see course/P5-cuda-deep-dive/CMakeLists.txt. Run: ./build/p5/p5.1_01_hello_indices   (sm_75+)
// Expected: 2 blocks × (4×2) threads; global x/y follow blockIdx*blockDim+threadIdx; warp = linear thread id / 32.
#include <cstdio>

#include "s2s_cuda.cuh"

__device__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

__global__ void who_am_i() {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;      // x is the fastest-varying dimension
  printf("block (%d,%d) thread (%d,%d) → global (%d,%d)  linear-in-block %d  warp %d  lane %d  SM %u\n",
         blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y, linear, linear / 32, linear % 32, smid());
}

__global__ void block_to_sm(unsigned* sm_of_block) {
  if (threadIdx.x == 0) sm_of_block[blockIdx.x] = smid();
}

int main() {
  who_am_i<<<dim3(2, 1), dim3(4, 2)>>>();
  CUDA_CHECK_LAUNCH();
  const int blocks = 4 * []{ int n; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n; }();
  s2s::DeviceBuffer<unsigned> d(blocks);
  block_to_sm<<<blocks, 64>>>(d.get());
  CUDA_CHECK_LAUNCH();
  auto h = d.download();
  std::printf("\n%d blocks → SM ids of the first 16: ", blocks);
  for (int i = 0; i < 16 && i < blocks; ++i) std::printf("%u ", h[i]);
  std::printf("\n(The block scheduler places blocks on SMs as resources free up; never rely on the order.)\n");
}
