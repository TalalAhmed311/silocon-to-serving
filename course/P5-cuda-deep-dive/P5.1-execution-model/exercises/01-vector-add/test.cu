#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(long long n) {
  auto a = s2s::random_vec<float>(size_t(n), -1, 1, 1), b = s2s::random_vec<float>(size_t(n), -1, 1, 2);
  std::vector<float> want(size_t(n));
  for (long long i = 0; i < n; ++i) want[size_t(i)] = a[size_t(i)] + b[size_t(i)];
  s2s::DeviceBuffer<float> da(a), db(b), dc(size_t(n));
  dc.zero();
  vadd(da.get(), db.get(), dc.get(), n);
  CUDA_CHECK_LAUNCH();
  auto got = dc.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(any_n) { for (long long n : {1LL, 31LL, 32LL, 1000LL, 1LL << 20, (1LL << 24) + 3}) check(n); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const long long n = 1LL << 26;
  s2s::DeviceBuffer<float> a(size_t(n)), b(size_t(n)), c(size_t(n));
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { vadd(a.get(), b.get(), c.get(), n); });
  s2s::report("vadd", "yours", double(n), t, 12.0 * n / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs() * 1.5);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
