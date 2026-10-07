#include <cmath>
#include <cstring>

#include <cub/cub.cuh>
#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N) {
  auto in = s2s::random_vec<float>(size_t(N), -1, 1, unsigned(N));
  double want = 0, mag = 0;
  for (float x : in) { want += x; mag += std::fabs(x); }
  s2s::DeviceBuffer<float> din(in), dout(1);
  solve(din.get(), dout.get(), N);
  CUDA_CHECK_LAUNCH();
  const float got = dout.download()[0];
  CHECK_NEAR(got, want, 0.0, 1e-6 * mag + 1e-5);   // float atomics: order varies, so compare against the magnitude
}

S2S_TEST(sizes) { for (int n : {1, 3, 4, 31, 1000, 4097, 1 << 20, (1 << 22) + 7}) check(n); }

// Exit check for L3: within 10% of cub::DeviceReduce::Sum on 2^24..2^28 floats (median of 100 runs).
S2S_TEST(bench) {
  if (!g_bench) return;
  s2s::table_header("GB/s");
  const double copy = s2s::measure_copy_gbs() / 2;   // read-only kernel: compare with half the copy's (read+write) rate
  for (int lg : {24, 26, 28}) {
    const int N = 1 << lg;
    s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(N))), out(1);
    auto t = s2s::time_gpu([&] { solve(in.get(), out.get(), N); }, 5, 100);
    s2s::report("reduction", "ours", N, t, 4.0 * N / (t.median_ms * 1e6), "GB/s", copy);
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, tmp_bytes, in.get(), out.get(), N);
    s2s::DeviceBuffer<char> tmp(tmp_bytes);
    auto tc = s2s::time_gpu([&] { cub::DeviceReduce::Sum(tmp.get(), tmp_bytes, in.get(), out.get(), N); }, 5, 100);
    s2s::report("reduction", "cub::DeviceReduce::Sum", N, tc, 4.0 * N / (tc.median_ms * 1e6), "GB/s", copy);
    std::printf("ours / cub time ratio at 2^%d: %.2f (exit check: <= 1.10)\n", lg, t.median_ms / tc.median_ms);
  }
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
