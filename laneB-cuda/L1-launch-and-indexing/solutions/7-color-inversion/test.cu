#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(shapes) {
  for (auto wh : {std::pair{1, 1}, std::pair{3, 5}, std::pair{640, 480}, std::pair{1919, 1081}}) {
    const int n = wh.first * wh.second;
    auto img = s2s::random_vec<unsigned char>(size_t(4 * n), 0, 255, unsigned(n));
    s2s::DeviceBuffer<unsigned char> d(img);
    solve(d.get(), wh.first, wh.second);
    CUDA_CHECK_LAUNCH();
    auto got = d.download();
    bool ok = true;
    for (int i = 0; i < 4 * n; ++i) ok &= got[size_t(i)] == ((i % 4 == 3) ? img[size_t(i)] : 255 - img[size_t(i)]);
    CHECK(ok);
  }
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int w = 8192, h = 8192;
  s2s::DeviceBuffer<unsigned char> d(size_t(4) * w * h);
  d.zero();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(d.get(), w, h); });
  s2s::report("color_inversion", "uchar4 per thread", double(w) * h, t, 8.0 * w * h / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
