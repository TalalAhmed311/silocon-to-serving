// Three cases. The faulting case runs LAST: after a device-side fault the CUDA context is unusable, so every
// later call in this process would fail too. (A real program would exit, as CUDA_CHECK does.)
#include <vector>

#include <s2s/check.hpp>
#include <kernel.cuh>
#include "s2s_cuda.cuh"  // only for DeviceBuffer and CUDA_CHECK; the kernel.cuh functions return errors instead

S2S_TEST(valid_launch_reports_success_and_scales) {
  cudaGetLastError();  // clear any stale error
  std::vector<float> h(1000);
  for (size_t i = 0; i < h.size(); ++i) h[i] = float(i);
  s2s::DeviceBuffer<float> d(h);
  CHECK(launch_scale(d.get(), 1000, 256) == cudaSuccess);
  auto got = d.download();
  for (size_t i = 0; i < h.size(); ++i) CHECK_NEAR(got[i], 2.0 * h[i], 0.0, 0.0);
}

S2S_TEST(too_many_threads_per_block_is_reported) {
  cudaGetLastError();
  s2s::DeviceBuffer<float> d(size_t(4096));
  d.zero();
  cudaError_t e = launch_scale(d.get(), 4096, 2048);  // 2048 > 1024: invalid configuration
  CHECK(e == cudaErrorInvalidConfiguration);
  cudaGetLastError();  // leave a clean slate for the next case
}

S2S_TEST(fault_inside_the_kernel_is_reported_at_the_sync) {
  cudaGetLastError();
  cudaError_t e = launch_fault();
  CHECK(e != cudaSuccess);  // typically cudaErrorIllegalAddress
}

S2S_TEST_MAIN()
