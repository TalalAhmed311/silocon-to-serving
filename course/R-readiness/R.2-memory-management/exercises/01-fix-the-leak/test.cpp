// Counts live heap blocks by replacing the global operator new / delete.
#include <cstdlib>
#include <new>
#include <s2s/check.hpp>
#include <impl.hpp>

static long g_live = 0;  // allocations minus frees, since program start

void* operator new(std::size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  ++g_live;
  return p;
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { if (p) { --g_live; std::free(p); } }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { operator delete(p); }

S2S_TEST(normal_path_gives_the_sum) {
  CHECK_EQ(sum_of_squares_below(4, 1000), 30L);  // 1 + 4 + 9 + 16
}

S2S_TEST(early_return_gives_minus_one) {
  CHECK_EQ(sum_of_squares_below(10, 50), -1L);  // 8*8 = 64 > 50
}

S2S_TEST(normal_path_frees_its_array) {
  const long before = g_live;
  sum_of_squares_below(100, 1000000);
  CHECK_EQ(g_live, before);
}

S2S_TEST(early_return_path_frees_its_array) {
  const long before = g_live;
  long r = sum_of_squares_below(100, 10);
  CHECK_EQ(r, -1L);
  CHECK_EQ(g_live, before);  // fails while the leak is still there
}

S2S_TEST_MAIN()
