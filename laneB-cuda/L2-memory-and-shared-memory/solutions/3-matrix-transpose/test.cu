#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(shapes) {
  for (auto rc : {std::pair{1, 1}, std::pair{1, 33}, std::pair{33, 1}, std::pair{31, 65}, std::pair{1000, 777}, std::pair{2048, 2048}}) {
    const int R = rc.first, C = rc.second;
    auto a = s2s::random_vec<float>(size_t(R) * C, -1, 1, unsigned(R * 7 + C));
    s2s::DeviceBuffer<float> din(a), dout(size_t(R) * C);
    solve(din.get(), dout.get(), R, C);
    CUDA_CHECK_LAUNCH();
    auto got = dout.download();
    std::vector<float> want(a.size());
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) want[size_t(c) * R + r] = a[size_t(r) * C + c];
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
  }
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 8192;
  s2s::DeviceBuffer<float> a(size_t(n) * n), b(size_t(n) * n);
  a.zero();
  const double copy = s2s::measure_copy_gbs(), gb = 8.0 * n * n;
  dim3 blk(TILE, ROWS_PER_PASS), grd(n / TILE, n / TILE), nb(32, 8), ng(n / 32, n / 8);
  s2s::table_header("GB/s");
  auto t0 = s2s::time_gpu([&] { transpose_naive<<<ng, nb>>>(a.get(), b.get(), n, n); });
  s2s::report("transpose", "naive", n, t0, gb / (t0.median_ms * 1e6), "GB/s", copy);
  auto t1 = s2s::time_gpu([&] { transpose_tiled<0><<<grd, blk>>>(a.get(), b.get(), n, n); });
  s2s::report("transpose", "smem, no pad (bank conflicts)", n, t1, gb / (t1.median_ms * 1e6), "GB/s", copy);
  auto t2 = s2s::time_gpu([&] { solve(a.get(), b.get(), n, n); });
  s2s::report("transpose", "smem + pad (exit check: >=80%)", n, t2, gb / (t2.median_ms * 1e6), "GB/s", copy);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
