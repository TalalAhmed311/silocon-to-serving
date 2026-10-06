// 05_templates.cpp — a strided MatrixView<T>, a constexpr tile chooser, and a concept.
// Run:      ./build/examples/05_templates
// Expected: prints the tile size chosen at compile time and a transposed view's elements.
// Hardware: T0.
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

// Concept: our numeric kernels accept arithmetic element types only, with a readable error otherwise.
template <class T>
concept Element = std::is_arithmetic_v<T>;

template <Element T>
struct MatrixView {
  T* data;
  int64_t rows, cols, stride0, stride1;
  T& operator()(int64_t i, int64_t j) const { return data[i * stride0 + j * stride1]; }
  MatrixView transposed() const { return {data, cols, rows, stride1, stride0}; }  // free: no bytes move
  bool is_contiguous() const { return stride1 == 1 && stride0 == cols; }
};

template <Element T>
MatrixView<T> row_major(std::span<T> buf, int64_t r, int64_t c) { return {buf.data(), r, c, c, 1}; }

// Choose a square tile so that two tiles (src + dst) fit in a 32 KB L1 data cache.
// constexpr: evaluated at compile time, so a bad choice is a compile error (see static_assert).
template <Element T>
constexpr int tile_for_l1(int l1_bytes = 32 * 1024) {
  int t = 1;
  while (2 * (2 * t) * (2 * t) * int(sizeof(T)) <= l1_bytes) t *= 2;
  return t;
}
static_assert(tile_for_l1<float>() == 64, "2 * 64*64*4 B = 32 KB");
static_assert(tile_for_l1<int8_t>() == 128, "2 * 128*128*1 B = 32 KB");

int main() {
  std::vector<float> buf(6);
  for (int i = 0; i < 6; ++i) buf[i] = float(i);
  auto m = row_major<float>(buf, 2, 3);
  auto t = m.transposed();
  std::printf("tile<float>=%d tile<int8>=%d\n", tile_for_l1<float>(), tile_for_l1<int8_t>());
  std::printf("m contiguous=%d, t contiguous=%d\n", m.is_contiguous(), t.is_contiguous());
  for (int64_t i = 0; i < t.rows; ++i) {
    for (int64_t j = 0; j < t.cols; ++j) std::printf("%4.0f", t(i, j));
    std::printf("\n");
  }
  // MatrixView<std::string> would fail with: "constraints not satisfied ... Element".
  return 0;
}
