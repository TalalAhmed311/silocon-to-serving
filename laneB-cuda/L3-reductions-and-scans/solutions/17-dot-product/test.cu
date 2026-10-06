#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N) {
  auto a = s2s::random_vec<float>(size_t(N), -1, 1, 1), b = s2s::random_vec<float>(size_t(N), -1, 1, 2);
  double want = 0, mag = 0;
  for (int i = 0; i < N; ++i) { want += double(a[size_t(i)]) * b[size_t(i)]; mag += std::fabs(double(a[size_t(i)]) * b[size_t(i)]); }
  s2s::DeviceBuffer<float> da(a), db(b), dr(1);
  solve(da.get(), db.get(), dr.get(), N);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(dr.download()[0], want, 0.0, 1e-6 * mag + 1e-5);
}

S2S_TEST(sizes) { for (int n : {1, 5, 1024, 100003, 1 << 22}) check(n); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 26;
  s2s::DeviceBuffer<float> a(s2s::random_vec<float>(size_t(N))), b(s2s::random_vec<float>(size_t(N), -1, 1, 3)), r(1);
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(a.get(), b.get(), r.get(), N); });
  s2s::report("dot", "fused float4 + shuffles", N, t, 8.0 * N / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs() / 2);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
