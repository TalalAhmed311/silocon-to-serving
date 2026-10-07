#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int C, int S, int G) {
  auto x = s2s::random_vec<float>(size_t(N) * C * S, -2, 3, 1);
  auto g = s2s::random_vec<float>(size_t(C), 0.5, 1.5, 2), b = s2s::random_vec<float>(size_t(C), -1, 1, 3);
  std::vector<float> want(x.size());
  const int cpg = C / G;
  for (int n = 0; n < N; ++n)
    for (int gr = 0; gr < G; ++gr) {
      const size_t base = (size_t(n) * C + size_t(gr) * cpg) * S, len = size_t(cpg) * S;
      double mu = 0, var = 0;
      for (size_t i = 0; i < len; ++i) mu += x[base + i];
      mu /= len;
      for (size_t i = 0; i < len; ++i) var += (x[base + i] - mu) * (x[base + i] - mu);
      var /= len;
      for (size_t i = 0; i < len; ++i) {
        const int c = gr * cpg + int(i / size_t(S));
        want[base + i] = float((x[base + i] - mu) / std::sqrt(var + 1e-5) * g[size_t(c)] + b[size_t(c)]);
      }
    }
  s2s::DeviceBuffer<float> dx(x), dg(g), db(b), dy(x.size());
  solve(dx.get(), dg.get(), db.get(), dy.get(), N, C, S, G, 1e-5f);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(shapes) { check(1, 4, 1, 2); check(2, 32, 49, 8); check(4, 64, 1024, 32); check(2, 6, 10, 1); }

S2S_TEST_MAIN()
