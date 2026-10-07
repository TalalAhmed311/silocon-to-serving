// 03_scan.cu — 3-phase scan-then-propagate vs single-pass decoupled look-back vs cub::DeviceScan (sm_75+).
// Expected: 3-phase moves ~4N floats of traffic (read, write, read+write for the carry), look-back ~2N, so look-back
// approaches CUB; GB/s below counts 2N (the minimum) for all three.
#include <cstdio>

#include <cub/cub.cuh>
#include <d4/scan.cuh>
#include "s2s_cuda.cuh"

int main() {
  const int n = 1 << 26, tiles = d4::ceil_div(n, d4::SCAN_TILE_N);
  s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(n))), out(size_t(n));
  s2s::DeviceBuffer<unsigned long long> status(size_t(tiles));
  s2s::DeviceBuffer<unsigned> counter(1);
  const double copy = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  auto t1 = s2s::time_gpu([&] { d4::scan_3phase(in.get(), out.get(), n); });
  s2s::report("scan", "3-phase", n, t1, 8.0 * n / (t1.median_ms * 1e6), "GB/s", copy);
  auto t2 = s2s::time_gpu([&] { d4::scan_lookback(in.get(), out.get(), n, status.get(), counter.get()); });
  s2s::report("scan", "decoupled look-back", n, t2, 8.0 * n / (t2.median_ms * 1e6), "GB/s", copy);
  size_t tmp = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp, in.get(), out.get(), n);
  s2s::DeviceBuffer<char> ws(tmp);
  auto t3 = s2s::time_gpu([&] { cub::DeviceScan::InclusiveSum(ws.get(), tmp, in.get(), out.get(), n); });
  s2s::report("scan", "cub::DeviceScan::InclusiveSum", n, t3, 8.0 * n / (t3.median_ms * 1e6), "GB/s", copy);
}
