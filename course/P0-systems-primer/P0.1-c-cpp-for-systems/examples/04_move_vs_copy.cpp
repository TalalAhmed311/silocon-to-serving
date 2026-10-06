// 04_move_vs_copy.cpp — the same 4-stage pipeline with copies vs moves, timed.
// Run:      ./build/examples/04_move_vs_copy
// Expected: the copy pipeline is far slower; the ratio grows with the buffer size.
// Hardware: T0. Numbers depend on your memory bandwidth — TODO(run): record yours in the lesson.
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include <s2s/bench.hpp>

using Tensor = std::vector<float>;

// By value: each call copies the whole tensor in, and the return is elided or moved out.
static Tensor stage_by_value(Tensor t) { t[0] += 1.0f; return t; }

static Tensor pipeline_copy(const Tensor& x) {
  Tensor a = stage_by_value(x);  // copy (x is const&, so it must copy)
  Tensor b = stage_by_value(a);  // copy (a is an lvalue)
  Tensor c = stage_by_value(b);  // copy
  return stage_by_value(c);      // copy
}

static Tensor pipeline_move(Tensor x) {
  Tensor a = stage_by_value(std::move(x));  // move: a pointer swap
  Tensor b = stage_by_value(std::move(a));
  Tensor c = stage_by_value(std::move(b));
  return stage_by_value(std::move(c));
}

int main() {
  const size_t n = 16u << 20;  // 16M floats = 64 MB
  Tensor src(n);
  std::iota(src.begin(), src.end(), 0.0f);

  auto tc = s2s::time_fn([&] { auto r = pipeline_copy(src); s2s::do_not_optimize(r[0]); }, 1, 7);
  // The move pipeline consumes its input, so give it a fresh copy *outside* the timed region... except
  // time_fn times the whole lambda; we subtract a measured copy-only baseline instead.
  auto tbase = s2s::time_fn([&] { Tensor t = src; s2s::do_not_optimize(t[0]); }, 1, 7);
  auto tm = s2s::time_fn([&] { Tensor t = src; auto r = pipeline_move(std::move(t)); s2s::do_not_optimize(r[0]); }, 1, 7);

  double move_only = tm.median_ms - tbase.median_ms;
  std::printf("| pipeline | median ms |\n|---|---|\n");
  std::printf("| 4 copies | %.2f |\n", tc.median_ms);
  std::printf("| 4 moves (after subtracting one setup copy) | %.3f |\n", move_only < 0 ? 0.0 : move_only);
  std::printf("| one 64 MB copy (baseline) | %.2f |\n", tbase.median_ms);
  return 0;
}
