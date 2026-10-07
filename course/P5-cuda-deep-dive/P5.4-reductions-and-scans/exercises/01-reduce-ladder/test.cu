#include <cmath>
#include <cstring>

#include <cub/cub.cuh>
#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int rung, long long n) {
  auto h = s2s::random_vec<float>(size_t(n), -1, 1, unsigned(n + rung));
  double want = 0, mag = 0;
  for (float x : h) { want += x; mag += std::fabs(x); }
  s2s::DeviceBuffer<float> in(h), out(1), partial(size_t(n / 256 + 2));
  out.zero();
  reduce_sum(rung, in.get(), out.get(), partial.get(), n);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(out.download()[0], want, 0.0, 1e-6 * mag + 1e-5);
}

S2S_TEST(every_rung) {
  for (int r = 1; r <= 6; ++r)
    for (long long n : {1LL, 300LL, 512LL, 513LL, 1LL << 20, (1LL << 22) + 5}) check(r, n);
}

// Exercise 2 — the L3 exit check: rung 6 within 10% of cub::DeviceReduce::Sum on 2^24..2^28, median of 100 runs.
S2S_TEST(bench) {
  if (!g_bench) return;
  for (int lg : {24, 26, 28}) {
    const long long n = 1LL << lg;
    s2s::DeviceBuffer<float> in(size_t(n)), out(1), partial(size_t(n / 256 + 2));
    in.zero();
    s2s::table_header("GB/s");
    s2s::GpuTiming best{};
    for (int r = 1; r <= 6; ++r) {
      auto t = s2s::time_gpu([&] { reduce_sum(r, in.get(), out.get(), partial.get(), n); }, 5, 100);
      s2s::report("reduce", "rung " + std::to_string(r), double(n), t, 4.0 * n / (t.median_ms * 1e6), "GB/s", 0);
      best = t;
    }
    size_t tmp = 0;
    cub::DeviceReduce::Sum(nullptr, tmp, in.get(), out.get(), int(n));
    s2s::DeviceBuffer<char> ws(tmp);
    auto tc = s2s::time_gpu([&] { cub::DeviceReduce::Sum(ws.get(), tmp, in.get(), out.get(), int(n)); }, 5, 100);
    s2s::report("reduce", "cub", double(n), tc, 4.0 * n / (tc.median_ms * 1e6), "GB/s", 0);
    std::printf("2^%d: rung 6 / cub = %.3f → exit check (<= 1.10): %s\n", lg, best.median_ms / tc.median_ms,
                best.median_ms <= 1.10 * tc.median_ms ? "PASS" : "not yet");
  }
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
