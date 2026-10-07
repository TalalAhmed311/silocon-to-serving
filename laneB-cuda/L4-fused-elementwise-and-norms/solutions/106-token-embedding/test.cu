#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int T, int D, int V) {
  auto table = s2s::random_vec<float>(size_t(V) * D, -1, 1, 1);
  auto ids = s2s::random_vec<int>(size_t(T), 0, V - 1, 2);
  if (T > 2) ids[1] = ids[0];                                 // repeated tokens
  std::vector<float> want(size_t(T) * D);
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < D; ++d) want[size_t(t) * D + d] = table[size_t(ids[size_t(t)]) * D + d];
  s2s::DeviceBuffer<float> dt(table), out(want.size());
  s2s::DeviceBuffer<int> di(ids);
  solve(di.get(), dt.get(), out.get(), T, D, V);
  CUDA_CHECK_LAUNCH();
  auto got = out.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(shapes) { check(1, 4, 10); check(17, 4096, 1000); check(300, 37, 50); }

S2S_TEST_MAIN()
