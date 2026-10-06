// transpose_bench.cpp — copy vs naive transpose vs your blocked transpose (exercise 4).
// Run:      ./build/bench/transpose_bench [N ...]     (default N = 1024 2048 4096)
// Output:   Markdown table + results/transpose.json. "% of peak" is relative to the copy row of the same N.
// Hardware: T0. Bytes moved per call = 2 * N * N * 4 (read src + write dst).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include <s2s/aligned.hpp>
#include <s2s/bench.hpp>
#include "impl.hpp"  // your transpose<T, Block>

static void naive(const float* s, float* d, int64_t r, int64_t c) {
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j) d[j * r + i] = s[i * c + j];
}

int main(int argc, char** argv) {
  std::vector<int64_t> sizes = {1024, 2048, 4096};
  if (argc > 1) { sizes.clear(); for (int i = 1; i < argc; ++i) sizes.push_back(std::atoll(argv[i])); }

  s2s::Table table("transpose", "GB/s");
  for (int64_t n : sizes) {
    s2s::aligned_buffer<float> src(size_t(n * n)), dst(size_t(n * n));
    for (int64_t k = 0; k < n * n; ++k) src[size_t(k)] = float(k);
    std::memset(dst.data(), 0, size_t(n * n) * sizeof(float));  // touch dst so page faults aren't timed
    const double gb = 2.0 * double(n) * double(n) * sizeof(float) / 1e9;

    auto t_copy = s2s::time_fn([&] { std::memcpy(dst.data(), src.data(), size_t(n * n) * sizeof(float)); s2s::do_not_optimize(dst[0]); });
    auto t_naive = s2s::time_fn([&] { naive(src.data(), dst.data(), n, n); s2s::do_not_optimize(dst[1]); });
    auto t_yours = s2s::time_fn([&] { transpose<float>(src.data(), dst.data(), n, n); s2s::do_not_optimize(dst[1]); });

    const double peak = gb / (t_copy.median_ms / 1e3);  // measured, not assumed
    table.add("copy (peak)", double(n), t_copy, peak, peak);
    table.add("naive transpose", double(n), t_naive, gb / (t_naive.median_ms / 1e3), peak);
    table.add("your transpose", double(n), t_yours, gb / (t_yours.median_ms / 1e3), peak);
  }
  table.print();
  std::filesystem::create_directories("results");
  table.save_json("results/transpose.json");
  std::printf("\nwrote results/transpose.json\n");
  return 0;
}
