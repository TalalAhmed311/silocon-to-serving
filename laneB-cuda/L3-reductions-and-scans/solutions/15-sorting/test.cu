#include <algorithm>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N) {
  auto in = s2s::random_vec<float>(size_t(N), -1000, 1000, unsigned(N));
  for (size_t i = 0; i < in.size(); i += 17) in[i] = 3.f;           // duplicates
  auto want = in;
  std::sort(want.begin(), want.end());
  s2s::DeviceBuffer<float> d(in);
  solve(d.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = d.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(sizes) { for (int n : {1, 2, 3, 1000, 2048, 2049, 100000, 1 << 20}) check(n); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 24;
  auto host = s2s::random_vec<float>(size_t(N));
  s2s::DeviceBuffer<float> d(host);
  s2s::table_header("Mkeys/s");
  auto t = s2s::time_gpu([&] { d.upload(host); solve(d.get(), N); }, 1, 10);   // includes the re-upload: compare relative only
  s2s::report("sort", "bitonic (smem stages for j < 2048)", N, t, N / (t.median_ms * 1e3), "Mkeys/s", 0);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
