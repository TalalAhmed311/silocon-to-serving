// 01_reduce_ladder.cu — the six rungs of d4/reduce.cuh plus CUB, at 2^24..2^26 floats (sm_75+).
// Expected: each rung faster than the last; rung 6 within ~10% of CUB (the L3 exit check); the "% of peak" column is
// against HALF the measured copy bandwidth (a reduction only reads).
#include <cstdio>

#include <cub/cub.cuh>
#include <d4/reduce.cuh>
#include "s2s_cuda.cuh"

int main() {
  const char* names[] = {"", "1 interleaved, divergent", "2 interleaved, strided index", "3 sequential addressing",
                         "4 first add during load", "5 + warp shuffles", "6 grid-stride float4 + 1 atomic"};
  const double peak = s2s::measure_copy_gbs() / 2;
  for (int lg : {24, 26}) {
    const long long n = 1LL << lg;
    s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(n))), out(1), partial(size_t(n / 256 + 1));
    s2s::table_header("GB/s");
    for (int r = 1; r <= 6; ++r) {
      auto t = s2s::time_gpu([&] { d4::reduce_sum(r, in.get(), out.get(), partial.get(), n); }, 5, 100);
      s2s::report("reduce_ladder", names[r], double(n), t, 4.0 * n / (t.median_ms * 1e6), "GB/s", peak);
    }
    size_t tmp = 0;
    cub::DeviceReduce::Sum(nullptr, tmp, in.get(), out.get(), int(n));
    s2s::DeviceBuffer<char> ws(tmp);
    auto tc = s2s::time_gpu([&] { cub::DeviceReduce::Sum(ws.get(), tmp, in.get(), out.get(), int(n)); }, 5, 100);
    s2s::report("reduce_ladder", "cub::DeviceReduce::Sum", double(n), tc, 4.0 * n / (tc.median_ms * 1e6), "GB/s", peak);
  }
}
