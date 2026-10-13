#include <type_traits>
#include <utility>
#include <vector>
#include <s2s/check.hpp>
#include <impl.hpp>

static_assert(!std::is_copy_constructible_v<FloatBuffer>, "copying must be deleted");
static_assert(!std::is_copy_assignable_v<FloatBuffer>, "copy assignment must be deleted");
static_assert(std::is_nothrow_move_constructible_v<FloatBuffer>, "mark the move constructor noexcept");
static_assert(std::is_nothrow_move_assignable_v<FloatBuffer>, "mark the move assignment noexcept");

S2S_TEST(constructor_allocates_zeroed_memory) {
  FloatBuffer b(8);
  CHECK_EQ(int(b.size()), 8);
  CHECK(b.data() != nullptr);
  CHECK_EQ(FloatBuffer::live_allocations(), 1);
  for (size_t i = 0; i < b.size(); ++i) CHECK_NEAR(b[i], 0.0, 0, 0);
}

S2S_TEST(destructor_frees) {
  CHECK_EQ(FloatBuffer::live_allocations(), 0);
  {
    FloatBuffer a(4);
    FloatBuffer b(16);
    CHECK_EQ(FloatBuffer::live_allocations(), 2);
  }
  CHECK_EQ(FloatBuffer::live_allocations(), 0);
}

S2S_TEST(move_construct_hands_over_the_pointer) {
  FloatBuffer a(5);
  a[2] = 7.5f;
  float* p = a.data();
  FloatBuffer b(std::move(a));
  CHECK(b.data() == p);          // same block, no copy
  CHECK_NEAR(b[2], 7.5, 0, 0);
  CHECK(a.data() == nullptr);    // the moved-from object is empty
  CHECK_EQ(int(a.size()), 0);
  CHECK_EQ(FloatBuffer::live_allocations(), 1);
}

S2S_TEST(move_assign_frees_the_old_block) {
  FloatBuffer a(3), b(9);
  CHECK_EQ(FloatBuffer::live_allocations(), 2);
  b = std::move(a);
  CHECK_EQ(int(b.size()), 3);
  CHECK_EQ(FloatBuffer::live_allocations(), 1);  // b's old 9-float block was freed
  CHECK(a.data() == nullptr);
}

S2S_TEST(self_move_assign_is_safe) {
  FloatBuffer a(4);
  FloatBuffer& same = a;
  a = std::move(same);
  CHECK_EQ(int(a.size()), 4);
  CHECK_EQ(FloatBuffer::live_allocations(), 1);
}

S2S_TEST(vector_growth_moves_instead_of_copying) {
  {
    std::vector<FloatBuffer> v;
    for (int i = 0; i < 20; ++i) v.emplace_back(10);
    CHECK_EQ(FloatBuffer::live_allocations(), 20);
    CHECK_EQ(int(v[0].size()), 10);
  }
  CHECK_EQ(FloatBuffer::live_allocations(), 0);
}

S2S_TEST(empty_buffer_owns_nothing) {
  FloatBuffer e(0);
  CHECK(e.data() == nullptr);
  CHECK_EQ(FloatBuffer::live_allocations(), 0);
}

S2S_TEST_MAIN()
