// 01_strided_read.cu — effective read bandwidth vs access stride (sm_75+). Each warp reads 32 floats spaced `stride`
// elements apart; useful bytes per 32-byte sector fetched drop from 32/32 (stride 1) to 4/32 (stride ≥ 8).
// Expected: GB/s of USEFUL data falls ~2× per stride doubling until stride 8, then flattens (one sector per lane).
// Also a misaligned case (offset 1 float): a warp's 128 contiguous bytes straddle 5 sectors instead of 4.
#include <cstdio>

#include "s2s_cuda.cuh"

__global__ void strided_read(const float* __restrict__ in, float* __restrict__ out, long long n_useful, int stride, int offset) {
  float acc = 0.f;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n_useful; i += (long long)gridDim.x * blockDim.x)
    acc += in[i * stride + offset];
  if (acc == 12345.f) out[0] = acc;   // keep the loads alive without writing anything else
}

int main() {
  const long long useful = 1 << 24;                      // 64 MB of useful floats per run
  s2s::DeviceBuffer<float> in(size_t(useful) * 32 + 64), out(1);
  in.zero();
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  std::printf("| stride | offset | useful GB/s | sectors per warp request (ideal) |\n|---|---|---|---|\n");
  for (int stride : {1, 2, 4, 8, 16, 32}) {
    for (int offset : {0, 1}) {
      if (offset && stride != 1) continue;
      auto t = s2s::time_gpu([&] { strided_read<<<sms * 8, 256>>>(in.get(), out.get(), useful, stride, offset); });
      const int sectors = stride >= 8 ? 32 : 4 * stride + (offset ? 1 : 0);
      std::printf("| %d | %d | %.1f | %d |\n", stride, offset, 4.0 * useful / (t.median_ms * 1e6), sectors);
    }
  }
  std::printf("Confirm the sector counts with: ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
              "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./p5.2_01_strided_read\n");
}
