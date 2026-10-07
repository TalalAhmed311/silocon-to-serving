#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N, int bins, bool skewed) {
  auto in = s2s::random_vec<int>(size_t(N), 0, bins - 1, unsigned(N + bins));
  if (skewed) for (size_t i = 0; i < in.size(); i += 2) in[i] = 0;   // half the data in bin 0
  s2s::DeviceBuffer<int> din(in), dh(size_t(bins));
  solve(din.get(), dh.get(), N, bins);
  CUDA_CHECK_LAUNCH();
  auto got = dh.download();
  std::vector<int> want(size_t(bins), 0);
  for (int v : in) want[size_t(v)]++;
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(uniform) { for (int bins : {1, 10, 256, 4096}) check(1 << 20, bins, false); }
S2S_TEST(skewed) { check(1 << 20, 256, true); }
S2S_TEST(many_bins_fallback) { check(100000, 20000, false); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 26, bins = 256;
  s2s::DeviceBuffer<int> in(s2s::random_vec<int>(size_t(N), 0, bins - 1)), h(size_t(bins));
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(in.get(), h.get(), N, bins); });
  s2s::report("histogram", "privatized smem", N, t, 4.0 * N / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs() / 2);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
