#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(counts) {
  for (int n : {1, 31, 1000, 1 << 22}) {
    auto a = s2s::random_vec<int>(size_t(n), 0, 9, unsigned(n));
    s2s::DeviceBuffer<int> da(a), dout(1);
    solve(da.get(), dout.get(), n, 3);
    CUDA_CHECK_LAUNCH();
    int want = 0;
    for (int v : a) want += v == 3;
    CHECK_EQ(dout.download()[0], want);
  }
}

int main() { return s2s::run_all(); }
