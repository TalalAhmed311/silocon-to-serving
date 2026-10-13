// Smoke test: the launch succeeds (CUDA_CHECK_LAUNCH aborts with file:line if it does not) and every thread
// recorded its own global index. Printed lines go to stdout; the test does not capture them.
#include <vector>

#include <s2s/check.hpp>
#include <kernel.cuh>
#include "s2s_cuda.cuh"

S2S_TEST(two_blocks_of_four) {
  std::vector<int> init(8, -1);
  s2s::DeviceBuffer<int> d(init);
  launch_hello(d.get());
  CUDA_CHECK_LAUNCH();  // cudaGetLastError() + cudaDeviceSynchronize(); this also flushes the kernel's printf
  auto got = d.download();
  for (int i = 0; i < 8; ++i) CHECK_EQ(got[size_t(i)], i);
}

S2S_TEST_MAIN()
