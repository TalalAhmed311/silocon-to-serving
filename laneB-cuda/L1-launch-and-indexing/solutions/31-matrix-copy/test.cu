#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(shapes) {
  for (int n : {1, 15, 32, 33, 1000}) {
    auto a = s2s::random_vec<float>(size_t(n) * n, -1, 1, unsigned(n));
    s2s::DeviceBuffer<float> da(a), db(size_t(n) * n);
    solve(da.get(), db.get(), n);
    CUDA_CHECK_LAUNCH();
    auto b = db.download();
    CHECK_ALLCLOSE(b.data(), a.data(), a.size(), 0.0, 0.0);
  }
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 8192;
  s2s::DeviceBuffer<float> a(size_t(n) * n), b(size_t(n) * n);
  a.zero();
  const double peak = s2s::measure_copy_gbs(), gb = 8.0 * n * n;
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(a.get(), b.get(), n); });
  s2s::report("matrix_copy", "x->col (coalesced)", n, t, gb / (t.median_ms * 1e6), "GB/s", peak);
  dim3 block(32, 8), grid((n + 31) / 32, (n + 7) / 8);
  auto t2 = s2s::time_gpu([&] { copy2d_uncoalesced<<<grid, block>>>(a.get(), b.get(), n); });
  s2s::report("matrix_copy", "x->row (uncoalesced)", n, t2, gb / (t2.median_ms * 1e6), "GB/s", peak);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
