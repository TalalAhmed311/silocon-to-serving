#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(sizes) {
  for (int n : {1, 33, 1 << 20}) {
    auto a = s2s::random_vec<float>(size_t(n), -1, 1, 1), b = s2s::random_vec<float>(size_t(n), -1, 1, 2);
    s2s::DeviceBuffer<float> da(a), db(b), dout(size_t(2 * n));
    solve(da.get(), db.get(), dout.get(), n);
    CUDA_CHECK_LAUNCH();
    auto out = dout.download();
    std::vector<float> want(size_t(2 * n));
    for (int i = 0; i < n; ++i) { want[size_t(2 * i)] = a[size_t(i)]; want[size_t(2 * i + 1)] = b[size_t(i)]; }
    CHECK_ALLCLOSE(out.data(), want.data(), want.size(), 0.0, 0.0);
  }
}

int main() { return s2s::run_all(); }
