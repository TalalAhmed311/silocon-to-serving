#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(shapes) {
  for (int n : {1, 17, 256, 2049}) {
    auto a = s2s::random_vec<float>(size_t(n) * n, -1, 1, 1), b = s2s::random_vec<float>(size_t(n) * n, -1, 1, 2);
    s2s::DeviceBuffer<float> da(a), db(b), dc(size_t(n) * n);
    solve(da.get(), db.get(), dc.get(), n);
    CUDA_CHECK_LAUNCH();
    auto c = dc.download();
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    CHECK_ALLCLOSE(c.data(), a.data(), a.size(), 0.0, 0.0);
  }
}

int main() { return s2s::run_all(); }
