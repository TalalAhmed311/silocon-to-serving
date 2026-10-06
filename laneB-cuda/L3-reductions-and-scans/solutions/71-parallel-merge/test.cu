#include <algorithm>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int m, int n, int vmax) {
  auto ai = s2s::random_vec<int>(size_t(m), 0, vmax, unsigned(m + 1)), bi = s2s::random_vec<int>(size_t(n), 0, vmax, unsigned(n + 2));
  std::vector<float> a(ai.begin(), ai.end()), b(bi.begin(), bi.end()), want(size_t(m + n));
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), want.begin());
  s2s::DeviceBuffer<float> da(a), db(b), dc(size_t(m + n));
  solve(da.get(), db.get(), dc.get(), m, n);
  CUDA_CHECK_LAUNCH();
  auto got = dc.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(shapes) {
  check(1, 0, 10);
  check(0, 5, 10);
  check(7, 3, 10);
  check(1000, 1, 1000);
  check(100000, 70001, 1 << 20);
  check(50000, 50000, 3);             // heavy ties
  check(1 << 20, 1 << 18, 1 << 30);
}

int main() { return s2s::run_all(); }
