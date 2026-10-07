#include <algorithm>
#include <cstring>

#include <cub/cub.cuh>
#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N, long long hi) {
  auto v = s2s::random_vec<long long>(size_t(N), 0, hi, unsigned(N));
  std::vector<unsigned> in(v.begin(), v.end());
  auto want = in;
  std::sort(want.begin(), want.end());
  s2s::DeviceBuffer<unsigned> d(in);
  solve(d.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = d.download();
  CHECK(std::equal(got.begin(), got.end(), want.begin()));
}

S2S_TEST(sizes) {
  for (int n : {1, 2, 2047, 2048, 2049, 100000}) check(n, 0xFFFFFFFFLL);
  check(1 << 20, 0xFFFFFFFFLL);
  check(1 << 20, 15);                 // many duplicates: stability bookkeeping under heavy ties
  check(5000, 0xFFFFFFFFLL);
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 24;
  auto v = s2s::random_vec<long long>(size_t(N), 0, 0xFFFFFFFFLL);
  std::vector<unsigned> host(v.begin(), v.end());
  s2s::DeviceBuffer<unsigned> d(host), o(size_t(N));
  s2s::table_header("Mkeys/s");
  auto t = s2s::time_gpu([&] { solve(d.get(), N); }, 1, 10);    // re-sorting sorted data costs the same for LSD radix
  s2s::report("radix_sort", "ours (8-bit LSD, 3-phase)", N, t, N / (t.median_ms * 1e3), "Mkeys/s", 0);
  size_t tmp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, tmp_bytes, d.get(), o.get(), N);
  s2s::DeviceBuffer<char> tmp(tmp_bytes);
  auto tc = s2s::time_gpu([&] { cub::DeviceRadixSort::SortKeys(tmp.get(), tmp_bytes, d.get(), o.get(), N); }, 1, 10);
  s2s::report("radix_sort", "cub::DeviceRadixSort", N, tc, N / (tc.median_ms * 1e3), "Mkeys/s", 0);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
