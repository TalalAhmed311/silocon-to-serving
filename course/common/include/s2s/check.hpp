// s2s/check.hpp — a tiny dependency-free test harness (no GoogleTest download needed).
// Usage:
//   #include <s2s/check.hpp>
//   S2S_TEST(my_test) { CHECK(1 + 1 == 2); CHECK_NEAR(x, 1.0, 1e-6, 0); }
//   S2S_TEST_MAIN()
// Each test binary returns non-zero if any check failed, so CTest reports it.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace s2s {

struct TestCase { const char* name; std::function<void()> fn; };

inline std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
inline int& failures() { static int f = 0; return f; }

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline void report_fail(const char* file, int line, const std::string& msg) {
  std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
  ++failures();
}

// |a - b| <= atol + rtol * |b|  — the same rule as numpy.allclose, so C++ and Python tests agree.
inline bool close(double a, double b, double rtol, double atol) {
  if (std::isnan(a) || std::isnan(b)) return false;
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// Compares two buffers; reports the first few mismatches instead of drowning the log.
template <class T, class U>
bool allclose(const T* got, const U* want, size_t n, double rtol, double atol,
              const char* file, int line) {
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!close(double(got[i]), double(want[i]), rtol, atol)) {
      if (bad < 5) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "index %zu: got %.9g want %.9g", i, double(got[i]), double(want[i]));
        report_fail(file, line, buf);
      }
      ++bad;
    }
  }
  if (bad > 5) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "... %zu mismatches total of %zu", bad, n);
    report_fail(file, line, buf);
  }
  return bad == 0;
}

inline int run_all() {
  int failed_tests = 0;
  for (auto& t : registry()) {
    int before = failures();
    t.fn();
    bool ok = failures() == before;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", t.name);
    failed_tests += !ok;
  }
  std::printf("%zu tests, %d failed\n", registry().size(), failed_tests);
  return failed_tests ? 1 : 0;
}

}  // namespace s2s

#define S2S_TEST(name)                                              \
  static void name();                                               \
  static ::s2s::Registrar name##_registrar(#name, name);            \
  static void name()
#define S2S_TEST_MAIN() int main() { return ::s2s::run_all(); }

#define CHECK(cond) \
  do { if (!(cond)) ::s2s::report_fail(__FILE__, __LINE__, "CHECK(" #cond ")"); } while (0)
#define CHECK_EQ(a, b)                                                                  \
  do { auto _a = (a); auto _b = (b);                                                    \
       if (!(_a == _b)) ::s2s::report_fail(__FILE__, __LINE__,                          \
           std::string(#a " == " #b " (got ") + std::to_string(_a) + " vs " + std::to_string(_b) + ")"); \
  } while (0)
#define CHECK_NEAR(a, b, rtol, atol)                                                    \
  do { double _a = (a), _b = (b);                                                       \
       if (!::s2s::close(_a, _b, rtol, atol)) ::s2s::report_fail(__FILE__, __LINE__,    \
           std::string(#a " ~= " #b " (got ") + std::to_string(_a) + " vs " + std::to_string(_b) + ")"); \
  } while (0)
#define CHECK_ALLCLOSE(got, want, n, rtol, atol) \
  ::s2s::allclose((got), (want), (n), (rtol), (atol), __FILE__, __LINE__)
