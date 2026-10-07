#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

static_assert(!std::is_copy_constructible_v<Tensor>, "Tensor must be move-only");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "move ctor must be noexcept (std::vector relies on it)");

S2S_TEST(aligned_and_zeroed) {
  Tensor t(3, 5);
  CHECK(t.data() != nullptr);
  CHECK(reinterpret_cast<uintptr_t>(t.data()) % 64 == 0);
  bool zero = true;
  for (int64_t i = 0; i < t.numel(); ++i) zero &= t.data()[i] == 0.0f;
  CHECK(zero);
  CHECK_EQ(t.numel(), int64_t(15));
}

S2S_TEST(move_construct_leaves_source_empty) {
  int base = Tensor::live_count();
  Tensor a(4, 4);
  a(1, 2) = 7.0f;
  Tensor b(std::move(a));
  CHECK(a.data() == nullptr);
  CHECK_EQ(a.rows(), int64_t(0));
  CHECK_EQ(b(1, 2), 7.0f);
  CHECK_EQ(Tensor::live_count(), base + 1);  // one allocation, one owner
}

S2S_TEST(move_assign_frees_old_storage) {
  int base = Tensor::live_count();
  {
    Tensor a(2, 2), b(8, 8);
    CHECK_EQ(Tensor::live_count(), base + 2);
    b = std::move(a);
    CHECK_EQ(Tensor::live_count(), base + 1);  // b's old 8x8 freed
    CHECK_EQ(b.rows(), int64_t(2));
  }
  CHECK_EQ(Tensor::live_count(), base);
}

S2S_TEST(self_move_is_safe) {
  Tensor a(2, 3);
  a(0, 0) = 1.0f;
  Tensor& alias = a;
  a = std::move(alias);
  CHECK(a.data() != nullptr);
  CHECK_EQ(a(0, 0), 1.0f);
}

S2S_TEST(clone_is_independent) {
  Tensor a(2, 2);
  a(0, 1) = 3.0f;
  Tensor c = a.clone();
  c(0, 1) = 9.0f;
  CHECK_EQ(a(0, 1), 3.0f);
  CHECK_EQ(c(0, 1), 9.0f);
  CHECK(c.data() != a.data());
}

S2S_TEST(vector_of_tensors_no_leaks) {
  int base = Tensor::live_count();
  {
    std::vector<Tensor> v;
    for (int i = 0; i < 100; ++i) v.emplace_back(1, i + 1);  // reallocations move, never copy
    CHECK_EQ(Tensor::live_count(), base + 100);
  }
  CHECK_EQ(Tensor::live_count(), base);
}

S2S_TEST_MAIN()
