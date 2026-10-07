// d4_bench.cu — one runner for every D4 kernel family, each with its baseline and its natural unit.
//   ./build/d4/d4_bench all | reduce | scan | softmax | rmsnorm | gemm | hgemm | attention
// Writes results/<family>.jsonl via s2s::report (one JSON line per row) for the site's charts.
#include <cstdio>
#include <cstring>
#include <string>

#include <cub/cub.cuh>
#include <d4/attention.cuh>
#include <d4/gemm.cuh>
#include <d4/hgemm.cuh>
#include <d4/norms.cuh>
#include <d4/reduce.cuh>
#include <d4/scan.cuh>
#include <d4/softmax.cuh>
#include "s2s_cuda.cuh"

static double g_copy = 0;

static void bench_reduce() {
  const long long n = 1LL << 26;
  s2s::DeviceBuffer<float> in(size_t(n)), out(1), partial(size_t(n / 256 + 1));
  in.zero();
  s2s::table_header("GB/s");
  for (int r : {3, 5, 6}) {
    auto t = s2s::time_gpu([&] { d4::reduce_sum(r, in.get(), out.get(), partial.get(), n); }, 5, 100);
    s2s::report("reduce", "rung " + std::to_string(r), double(n), t, 4.0 * n / (t.median_ms * 1e6), "GB/s", g_copy / 2);
  }
  size_t tmp = 0;
  cub::DeviceReduce::Sum(nullptr, tmp, in.get(), out.get(), int(n));
  s2s::DeviceBuffer<char> ws(tmp);
  auto t = s2s::time_gpu([&] { cub::DeviceReduce::Sum(ws.get(), tmp, in.get(), out.get(), int(n)); }, 5, 100);
  s2s::report("reduce", "cub", double(n), t, 4.0 * n / (t.median_ms * 1e6), "GB/s", g_copy / 2);
}

static void bench_scan() {
  const int n = 1 << 26, tiles = d4::ceil_div(n, d4::SCAN_TILE_N);
  s2s::DeviceBuffer<float> in(size_t(n)), out(size_t(n));
  s2s::DeviceBuffer<unsigned long long> st(size_t(tiles));
  s2s::DeviceBuffer<unsigned> ctr(1);
  in.zero();
  s2s::table_header("GB/s");
  auto t1 = s2s::time_gpu([&] { d4::scan_3phase(in.get(), out.get(), n); });
  s2s::report("scan", "3-phase", n, t1, 8.0 * n / (t1.median_ms * 1e6), "GB/s", g_copy);
  auto t2 = s2s::time_gpu([&] { d4::scan_lookback(in.get(), out.get(), n, st.get(), ctr.get()); });
  s2s::report("scan", "look-back", n, t2, 8.0 * n / (t2.median_ms * 1e6), "GB/s", g_copy);
}

static void bench_softmax() {
  const int rows = 65536, cols = 1024;
  s2s::DeviceBuffer<float> x(size_t(rows) * cols), y(size_t(rows) * cols);
  x.zero();
  s2s::table_header("GB/s");
  for (int v = 1; v <= 3; ++v) {
    auto t = s2s::time_gpu([&] { d4::softmax(v, x.get(), y.get(), rows, cols); });
    s2s::report("softmax", "v" + std::to_string(v), double(rows) * cols, t, 8.0 * rows * cols / (t.median_ms * 1e6), "GB/s", g_copy);
  }
}

static void bench_rmsnorm() {
  const int rows = 16384, cols = 4096;
  s2s::DeviceBuffer<__nv_bfloat16> x(size_t(rows) * cols), r(size_t(rows) * cols), w(size_t(cols)), y(size_t(rows) * cols);
  x.zero(); r.zero(); w.zero();
  s2s::table_header("GB/s");
  auto t1 = s2s::time_gpu([&] { d4::rmsnorm(x.get(), w.get(), y.get(), rows, cols); });
  s2s::report("rmsnorm", "bf16", double(rows) * cols, t1, 4.0 * rows * cols / (t1.median_ms * 1e6), "GB/s", g_copy);
  auto t2 = s2s::time_gpu([&] { d4::fused_add_rmsnorm(x.get(), r.get(), w.get(), y.get(), rows, cols); });
  s2s::report("rmsnorm", "fused add bf16", double(rows) * cols, t2, 8.0 * rows * cols / (t2.median_ms * 1e6), "GB/s", g_copy);
}

static void bench_gemm() {
  const int n = 4096;
  s2s::DeviceBuffer<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
  A.zero(); B.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  const double f = 2.0 * n * n * n;
  auto tc = s2s::time_gpu([&] { d4::gemm_cublas(h, n, n, n, A.get(), B.get(), C.get()); });
  s2s::table_header("TFLOP/s");
  s2s::report("gemm", "cuBLAS", n, tc, f / (tc.median_ms * 1e9), "TFLOP/s", f / (tc.median_ms * 1e9));
  for (int r = 3; r <= 7; ++r) {
    auto t = s2s::time_gpu([&] { d4::sgemm(r, n, n, n, A.get(), B.get(), C.get()); });
    s2s::report("gemm", "rung " + std::to_string(r), n, t, f / (t.median_ms * 1e9), "TFLOP/s", f / (tc.median_ms * 1e9));
  }
  cublasDestroy(h);
}

static void bench_hgemm() {
  const int n = 4096;
  s2s::DeviceBuffer<__half> A(size_t(n) * n), B(size_t(n) * n);
  s2s::DeviceBuffer<float> C(size_t(n) * n);
  A.zero(); B.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  const double f = 2.0 * n * n * n;
  auto tc = s2s::time_gpu([&] { d4::hgemm_cublas(h, n, n, n, A.get(), B.get(), C.get()); });
  s2s::table_header("TFLOP/s");
  s2s::report("hgemm", "cuBLAS", n, tc, f / (tc.median_ms * 1e9), "TFLOP/s", f / (tc.median_ms * 1e9));
  for (int v = 0; v <= 3; ++v) {
    if (!d4::hgemm_supported(v, n, n, n)) continue;
    auto t = s2s::time_gpu([&] { d4::hgemm(v, n, n, n, A.get(), B.get(), C.get()); });
    s2s::report("hgemm", "variant " + std::to_string(v), n, t, f / (t.median_ms * 1e9), "TFLOP/s", f / (tc.median_ms * 1e9));
  }
  cublasDestroy(h);
}

static void bench_attention() {
  const int BH = 16, N = 4096;
  const size_t n = size_t(BH) * N * d4::AD;
  s2s::DeviceBuffer<__half> Q(n), K(n), V(n), O(n);
  Q.zero(); K.zero(); V.zero();
  s2s::DeviceBuffer<float> S(size_t(BH) * N * N);
  const double f = 4.0 * BH * N * double(N) * d4::AD;
  s2s::table_header("TFLOP/s");
  auto tn = s2s::time_gpu([&] { d4::attn_naive(Q.get(), K.get(), V.get(), O.get(), S.get(), BH, N, false); }, 1, 5);
  s2s::report("attention", "naive", N, tn, f / (tn.median_ms * 1e9), "TFLOP/s", 0);
  auto t2 = s2s::time_gpu([&] { d4::attn_flash2(Q.get(), K.get(), V.get(), O.get(), BH, N, false); }, 2, 10);
  s2s::report("attention", "FA-2", N, t2, f / (t2.median_ms * 1e9), "TFLOP/s", 0);
  std::printf("FA-2 / naive speedup at N=%d: %.1fx (L6 exit check: >= 5x)\n", N, tn.median_ms / t2.median_ms);
}

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "all";
  std::printf("GPU: %s\n", s2s::gpu_name().c_str());
  g_copy = s2s::measure_copy_gbs();
  std::printf("measured copy bandwidth: %.1f GB/s (read + write)\n", g_copy);
  const std::pair<const char*, void (*)()> all[] = {{"reduce", bench_reduce}, {"scan", bench_scan}, {"softmax", bench_softmax},
                                                    {"rmsnorm", bench_rmsnorm}, {"gemm", bench_gemm}, {"hgemm", bench_hgemm},
                                                    {"attention", bench_attention}};
  for (auto& [name, fn] : all)
    if (which == "all" || which == name) { std::printf("\n## %s\n", name); fn(); }
}
