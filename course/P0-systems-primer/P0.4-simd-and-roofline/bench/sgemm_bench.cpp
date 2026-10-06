// sgemm_bench.cpp — every D1 rung at several N. Prints a Markdown table and results/sgemm.json.
// Run:      ./build/bench/sgemm_bench [peak_gflops]   (sgemm_bench.py passes the measured peak from results/peak.json)
// Hardware: T0. Naive is skipped above N=1024 (it would take minutes).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <s2s/aligned.hpp>
#include <s2s/bench.hpp>
#include "sgemm_threads.hpp"

int main(int argc, char** argv) {
  const double peak = argc > 1 ? std::atof(argv[1]) : 0.0;
  s2s::ThreadPool pool;
  s2s::Table t("sgemm", "GFLOP/s");
  for (int n : {256, 512, 1024, 2048}) {
    s2s::aligned_buffer<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
    for (size_t i = 0; i < A.size(); ++i) { A[i] = float(i % 7) * 0.1f; B[i] = float(i % 5) * 0.1f; C[i] = 0; }
    const double flops = 2.0 * n * n * n;
    auto add = [&](const std::string& label, auto fn, int reps) {
      auto tm = s2s::time_fn([&] { fn(); s2s::do_not_optimize(C[0]); }, 1, reps);
      t.add(label, n, tm, flops / (tm.median_ms * 1e6), peak);
    };
    if (n <= 1024) add("naive", [&] { d1::sgemm_naive(n, n, n, A.data(), B.data(), C.data()); }, 3);
    add("reorder", [&] { d1::sgemm_reorder(n, n, n, A.data(), B.data(), C.data()); }, 3);
    add("tiled", [&] { d1::sgemm_tiled(n, n, n, A.data(), B.data(), C.data()); }, 3);
    add("simd", [&] { d1::sgemm_simd(n, n, n, A.data(), B.data(), C.data()); }, 5);
    add("simd+threads", [&] { d1::sgemm_threads(n, n, n, A.data(), B.data(), C.data(), pool); }, 7);
  }
  t.print();
  std::filesystem::create_directories("results");
  t.save_json("results/sgemm.json");
  if (peak <= 0) std::printf("\n(%% of peak is 0: pass the measured peak GFLOP/s as argv[1], or use sgemm_bench.py)\n");
  return 0;
}
