#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(powers) {
  const int N = 33;
  auto a = s2s::random_vec<float>(size_t(N) * N, -0.15, 0.15, 3);   // small entries: A^P stays well-scaled
  for (int P : {0, 1, 2, 5, 16}) {
    s2s::DeviceBuffer<float> din(a), dout(size_t(N) * N);
    solve_power(din.get(), dout.get(), N, P);
    CUDA_CHECK_LAUNCH();
    std::vector<double> r(size_t(N) * N, 0.0), t(size_t(N) * N);
    for (int i = 0; i < N; ++i) r[size_t(i) * N + i] = 1;
    for (int k = 0; k < P; ++k) {
      std::fill(t.begin(), t.end(), 0.0);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          for (int x = 0; x < N; ++x) t[size_t(i) * N + j] += r[size_t(i) * N + x] * a[size_t(x) * N + j];
      r = t;
    }
    auto got = dout.download();
    CHECK_ALLCLOSE(got.data(), r.data(), r.size(), 1e-3, 1e-5);
  }
}

int main() { return s2s::run_all(); }
