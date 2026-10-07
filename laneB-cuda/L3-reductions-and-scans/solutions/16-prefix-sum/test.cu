#include <cmath>
#include <cstring>

#include <cub/cub.cuh>
#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

// Small integers stored as floats: every partial sum is exact (|sum| < 2^24), so the check can be exact too.
static void check_exact(int N) {
  auto in = s2s::random_vec<int>(size_t(N), -8, 8, unsigned(N));
  std::vector<float> f(in.begin(), in.end()), want(f.size());
  double run = 0;
  for (size_t i = 0; i < f.size(); ++i) { run += f[i]; want[i] = float(run); }
  s2s::DeviceBuffer<float> din(f), dout(f.size());
  solve(din.get(), dout.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(exact_sizes) {
  for (int n : {1, 7, 2047, 2048, 2049, 4096, 100000, 2048 * 2048 + 5}) check_exact(n);   // last one recurses twice
}

S2S_TEST(in_place) {
  const int N = 50000;
  std::vector<float> f(size_t(N), 1.f);
  s2s::DeviceBuffer<float> d(f);
  solve(d.get(), d.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = d.download();
  CHECK_EQ(got[0], 1.f);
  CHECK_EQ(got[size_t(N) - 1], float(N));
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 26;
  s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(N))), out(size_t(N));
  s2s::table_header("GB/s");
  const double copy = s2s::measure_copy_gbs();
  auto t = s2s::time_gpu([&] { solve(in.get(), out.get(), N); });
  s2s::report("scan", "3-phase tile scan", N, t, 8.0 * N / (t.median_ms * 1e6), "GB/s", copy);
  size_t tmp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, in.get(), out.get(), N);
  s2s::DeviceBuffer<char> tmp(tmp_bytes);
  auto tc = s2s::time_gpu([&] { cub::DeviceScan::InclusiveSum(tmp.get(), tmp_bytes, in.get(), out.get(), N); });
  s2s::report("scan", "cub::DeviceScan (decoupled look-back)", N, tc, 8.0 * N / (tc.median_ms * 1e6), "GB/s", copy);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
