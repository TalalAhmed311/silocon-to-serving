// 01_device_info.cu: what GPU do I have? Prints the fields the course keeps asking about.
// Build: see ../exercises/CMakeLists.txt (target p1.6_01_device_info). Needs an NVIDIA GPU. TODO(run-on: T2)
#include <cstdio>

#include "s2s_cuda.cuh"

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::printf("%d CUDA device(s)\n", count);
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    std::printf("device %d: %s\n", d, p.name);
    std::printf("  compute capability  %d.%d  (compile with -arch=sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    std::printf("  SMs                 %d\n", p.multiProcessorCount);
    std::printf("  global memory       %.1f GiB\n", double(p.totalGlobalMem) / double(1ull << 30));
    std::printf("  max threads/block   %d\n", p.maxThreadsPerBlock);
    std::printf("  warp size           %d\n", p.warpSize);
    std::printf("  L2 cache            %.1f MiB\n", double(p.l2CacheSize) / double(1 << 20));
  }
  return 0;
}
