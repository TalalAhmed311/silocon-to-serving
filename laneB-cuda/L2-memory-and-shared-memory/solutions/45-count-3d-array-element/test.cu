#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(counts) {
  const int D = 17, R = 129, C = 65;
  auto a = s2s::random_vec<int>(size_t(D) * R * C, 0, 4, 2);
  s2s::DeviceBuffer<int> da(a), dout(1);
  solve3d(da.get(), dout.get(), D, R, C, 1);
  CUDA_CHECK_LAUNCH();
  int want = 0;
  for (int v : a) want += v == 1;
  CHECK_EQ(dout.download()[0], want);
}

int main() { return s2s::run_all(); }
