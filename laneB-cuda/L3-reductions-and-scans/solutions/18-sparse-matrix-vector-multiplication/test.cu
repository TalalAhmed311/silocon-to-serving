#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

struct Csr { std::vector<int> rp, ci; std::vector<float> v; int rows, cols; };

static Csr random_csr(int rows, int cols, int max_per_row, unsigned seed) {
  Csr m{{0}, {}, {}, rows, cols};
  auto counts = s2s::random_vec<int>(size_t(rows), 0, max_per_row, seed);
  for (int r = 0; r < rows; ++r) {
    auto c = s2s::random_vec<int>(size_t(counts[size_t(r)]), 0, cols - 1, seed + unsigned(r) + 1);
    auto v = s2s::random_vec<float>(c.size(), -1, 1, seed + unsigned(r) + 7);
    m.ci.insert(m.ci.end(), c.begin(), c.end());   // duplicates in a row are allowed: they just add up
    m.v.insert(m.v.end(), v.begin(), v.end());
    m.rp.push_back(int(m.ci.size()));
  }
  return m;
}

static void check(int rows, int cols, int max_per_row) {
  auto m = random_csr(rows, cols, max_per_row, unsigned(rows + max_per_row));
  auto x = s2s::random_vec<float>(size_t(cols), -1, 1, 99);
  std::vector<float> want(size_t(rows));
  for (int r = 0; r < rows; ++r) {
    double acc = 0;
    for (int k = m.rp[size_t(r)]; k < m.rp[size_t(r) + 1]; ++k) acc += double(m.v[size_t(k)]) * x[size_t(m.ci[size_t(k)])];
    want[size_t(r)] = float(acc);
  }
  s2s::DeviceBuffer<int> rp(m.rp), ci(m.ci.empty() ? std::vector<int>{0} : m.ci);
  s2s::DeviceBuffer<float> v(m.v.empty() ? std::vector<float>{0} : m.v), dx(x), dy(size_t(rows));
  solve(rp.get(), ci.get(), v.get(), dx.get(), dy.get(), rows, int(m.ci.size()));
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-5);
}

S2S_TEST(short_rows) { check(1000, 500, 3); }          // thread-per-row path
S2S_TEST(long_rows) { check(5000, 4000, 200); }       // warp-per-row path
S2S_TEST(empty_rows) { check(100, 100, 0); }

S2S_TEST(bench) {
  if (!g_bench) return;
  auto m = random_csr(200000, 200000, 64, 5);
  s2s::DeviceBuffer<int> rp(m.rp), ci(m.ci);
  s2s::DeviceBuffer<float> v(m.v), x(s2s::random_vec<float>(200000)), y(200000);
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(rp.get(), ci.get(), v.get(), x.get(), y.get(), 200000, int(m.ci.size())); });
  const double bytes = 8.0 * m.ci.size() + 4.0 * 200001 + 8.0 * 200000;   // vals+cols, row_ptr, x (≈ once) + y
  s2s::report("spmv", "warp per row", double(m.ci.size()), t, bytes / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs() / 2);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
