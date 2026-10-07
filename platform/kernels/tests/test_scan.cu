#include <d4/scan.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

// Small integers as floats keep every partial sum exact, so the scans must match bit for bit.
static std::vector<float> ints(int n) {
  auto v = s2s::random_vec<int>(size_t(n), -8, 8, unsigned(n));
  return {v.begin(), v.end()};
}

static std::vector<float> ref(const std::vector<float>& x) {
  std::vector<float> y(x.size());
  double r = 0;
  for (size_t i = 0; i < x.size(); ++i) { r += x[i]; y[i] = float(r); }
  return y;
}

S2S_TEST(three_phase) {
  for (int n : {1, 2047, 2048, 2049, 100000, 2048 * 2048 + 3}) {
    auto x = ints(n), want = ref(x);
    s2s::DeviceBuffer<float> in(x), out(x.size());
    d4::scan_3phase(in.get(), out.get(), n);
    CUDA_CHECK_LAUNCH();
    auto got = out.download();
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
  }
}

S2S_TEST(decoupled_lookback) {
  for (int n : {1, 2048, 2049, 1 << 20, (1 << 22) + 11}) {
    auto x = ints(n), want = ref(x);
    const int tiles = d4::ceil_div(n, d4::SCAN_TILE_N);
    s2s::DeviceBuffer<float> in(x), out(x.size());
    s2s::DeviceBuffer<unsigned long long> status(size_t(tiles));
    s2s::DeviceBuffer<unsigned> counter(1);
    for (int rep = 0; rep < 3; ++rep) {     // races show up as flaky mismatches: run it a few times
      d4::scan_lookback(in.get(), out.get(), n, status.get(), counter.get());
      CUDA_CHECK_LAUNCH();
      auto got = out.download();
      CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
    }
  }
}

S2S_TEST_MAIN()
