#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(sizes) {
  for (int n : {1, 7, 256, 257, 1000, 1 << 20, (1 << 20) + 3}) {
    auto a = s2s::random_vec<float>(size_t(n), -1, 1, 1), b = s2s::random_vec<float>(size_t(n), -1, 1, 2);
    s2s::DeviceBuffer<float> da(a), db(b), dc(size_t(n));
    solve(da.get(), db.get(), dc.get(), n);
    CUDA_CHECK_LAUNCH();
    auto c = dc.download();
    std::vector<float> want(size_t(n));
    for (int i = 0; i < n; ++i) want[size_t(i)] = a[size_t(i)] + b[size_t(i)];
    CHECK_ALLCLOSE(c.data(), want.data(), want.size(), 0.0, 0.0);  // one IEEE add: bit-exact
  }
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 1 << 26;
  s2s::DeviceBuffer<float> a(size_t(n)), b(size_t(n)), c(size_t(n));
  a.zero(); b.zero();
  const double peak = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(a.get(), b.get(), c.get(), n); });
  s2s::report("vector_add", "1 thread/elem", n, t, 12.0 * n / (t.median_ms * 1e6), "GB/s", peak);
}

int main(int argc, char** argv) {
  g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  return s2s::run_all();
}
