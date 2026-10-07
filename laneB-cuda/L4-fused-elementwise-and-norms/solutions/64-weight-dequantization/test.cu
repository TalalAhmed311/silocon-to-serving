#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int rows, int cols, int group) {
  auto qi = s2s::random_vec<int>(size_t(rows) * cols, -128, 127, 1);
  std::vector<int8_t> q(qi.begin(), qi.end());
  auto sc = s2s::random_vec<float>(size_t(rows) * (cols / group), 0.001, 0.05, 2);
  std::vector<float> want(q.size());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) want[size_t(r) * cols + c] = float(q[size_t(r) * cols + c]) * sc[size_t(r) * (cols / group) + c / group];
  s2s::DeviceBuffer<int8_t> dq(q);
  s2s::DeviceBuffer<float> ds(sc), dw(want.size());
  solve(dq.get(), ds.get(), dw.get(), rows, cols, group);
  CUDA_CHECK_LAUNCH();
  auto got = dw.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);   // one fp32 multiply each side: exact
}

S2S_TEST(shapes) { check(1, 32, 32); check(7, 256, 64); check(512, 4096, 128); }

S2S_TEST_MAIN()
