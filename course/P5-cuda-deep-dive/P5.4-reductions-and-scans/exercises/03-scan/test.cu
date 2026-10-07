#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int n) {
  auto v = s2s::random_vec<int>(size_t(n), -8, 8, unsigned(n));
  std::vector<float> x(v.begin(), v.end()), want(x.size());
  double r = 0;
  for (size_t i = 0; i < x.size(); ++i) { r += x[i]; want[i] = float(r); }
  s2s::DeviceBuffer<float> in(x), out(x.size());
  inclusive_scan(in.get(), out.get(), n);
  CUDA_CHECK_LAUNCH();
  auto got = out.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);    // small ints as floats: exact
}

S2S_TEST(block_sized) { check(1); check(2047); check(2048); }
S2S_TEST(multi_block) { check(2049); check(1 << 20); check(2048 * 2048 + 1); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 1 << 26;
  s2s::DeviceBuffer<float> in(size_t(n)), out(size_t(n));
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { inclusive_scan(in.get(), out.get(), n); });
  s2s::report("scan", "yours", n, t, 8.0 * n / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
