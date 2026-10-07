#include <algorithm>
#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int rows, int vocab) {
  auto lg = s2s::random_vec<float>(size_t(rows) * vocab, -8, 8, unsigned(rows + vocab));
  auto tg = s2s::random_vec<int>(size_t(rows), 0, vocab - 1, 3);
  std::vector<float> want(size_t(rows));
  for (int r = 0; r < rows; ++r) {
    double m = -1e300, s = 0;
    for (int c = 0; c < vocab; ++c) m = std::max(m, double(lg[size_t(r) * vocab + c]));
    for (int c = 0; c < vocab; ++c) s += std::exp(lg[size_t(r) * vocab + c] - m);
    want[size_t(r)] = float(m + std::log(s) - lg[size_t(r) * vocab + size_t(tg[size_t(r)])]);
  }
  s2s::DeviceBuffer<float> dl(lg), dloss(size_t(rows));
  s2s::DeviceBuffer<int> dt(tg);
  cross_entropy(dl.get(), dt.get(), dloss.get(), rows, vocab);
  CUDA_CHECK_LAUNCH();
  auto got = dloss.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(vocab_sizes) { check(1, 2); check(7, 1000); check(64, 32000); check(8, 128256); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int rows = 4096, vocab = 128256;          // a Llama-3-sized vocabulary
  s2s::DeviceBuffer<float> lg(size_t(rows) * vocab), loss(size_t(rows));
  s2s::DeviceBuffer<int> t(size_t(rows));
  lg.zero(); t.zero();
  s2s::table_header("GB/s");
  auto tm = s2s::time_gpu([&] { cross_entropy(lg.get(), t.get(), loss.get(), rows, vocab); });
  s2s::report("cross_entropy", "fused", double(rows) * vocab, tm, 4.0 * rows * vocab / (tm.median_ms * 1e6), "GB/s",
              s2s::measure_copy_gbs() / 2);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
