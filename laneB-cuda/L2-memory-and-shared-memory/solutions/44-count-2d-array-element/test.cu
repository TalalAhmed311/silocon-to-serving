#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(counts) {
  const int R = 1023, C = 517;
  auto a = s2s::random_vec<int>(size_t(R) * C, 0, 4, 1);
  s2s::DeviceBuffer<int> da(a), dout(1);
  solve2d(da.get(), dout.get(), R, C, 2);
  CUDA_CHECK_LAUNCH();
  int want = 0;
  for (int v : a) want += v == 2;
  CHECK_EQ(dout.download()[0], want);
}

int main() { return s2s::run_all(); }
