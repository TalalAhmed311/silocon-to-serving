// 03_raii_buffer.cpp — count constructions, copies and moves to see what each line really does.
// Run:      ./build/examples/03_raii_buffer
// Expected: each line's counters as asserted below; the program aborts if they differ.
// Hardware: T0.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

struct Counters { int ctor = 0, copy = 0, move = 0, dtor_free = 0; };
static Counters C;

class Buffer {
 public:
  explicit Buffer(size_t n) : n_(n), p_(new float[n]()) { ++C.ctor; }
  ~Buffer() {
    if (p_) ++C.dtor_free;   // only count destructors that actually own memory
    delete[] p_;
  }
  // Deep copy: correct, but it costs n * 4 bytes of memory traffic. We keep it to make the cost visible.
  Buffer(const Buffer& o) : n_(o.n_), p_(new float[o.n_]) { std::memcpy(p_, o.p_, n_ * sizeof(float)); ++C.copy; }
  Buffer& operator=(const Buffer&) = delete;
  // Move: steal the pointer, leave the source empty so its destructor frees nothing.
  // noexcept matters: std::vector only moves elements on reallocation if the move can't throw.
  Buffer(Buffer&& o) noexcept : n_(std::exchange(o.n_, 0)), p_(std::exchange(o.p_, nullptr)) { ++C.move; }
  Buffer& operator=(Buffer&&) = delete;
  size_t size() const { return n_; }

 private:
  size_t n_;
  float* p_;
};

static void show(const char* what) {
  std::printf("%-22s-> ctor=%d copy=%d move=%d freed=%d\n", what, C.ctor, C.copy, C.move, C.dtor_free);
}

int main() {
  {
    Buffer a(1024);
    show("make a");
    assert(C.ctor == 1 && C.copy == 0 && C.move == 0);

    Buffer b = std::move(a);  // ownership moves; a is now empty
    show("b = std::move(a)");
    assert(C.move == 1 && a.size() == 0 && b.size() == 1024);

    Buffer c = b;  // a real copy: new allocation + memcpy
    show("c = b (copy)");
    assert(C.copy == 1);

    std::vector<Buffer> v;
    v.reserve(2);              // reserve so the push_backs below don't reallocate (that would add moves)
    v.push_back(std::move(c)); // move into the vector
    v.emplace_back(16);        // constructed in place: no copy, no move
    show("push_back + emplace");
    assert(C.ctor == 2 && C.move == 2 && C.copy == 1);
  }
  show("end of scope");
  // Owners alive at the end: b, v[0] (was c), v[1]. a was moved-from, so 3 frees, not 4.
  assert(C.dtor_free == 3);
  std::printf("all counter assertions passed\n");
  return 0;
}
