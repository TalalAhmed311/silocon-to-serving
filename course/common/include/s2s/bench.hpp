// s2s/bench.hpp — benchmark timing with warm-up, repeats, median and p90.
// Why median/p90 and not mean: a single OS hiccup can double a mean; the median is robust,
// and p90 tells you how noisy the machine is.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace s2s {

struct Timing { double median_ms, p90_ms, min_ms; };

// Runs fn `warmup` times untimed (caches, page faults, CPU frequency ramp), then `reps` timed.
inline Timing time_fn(const std::function<void()>& fn, int warmup = 3, int reps = 15) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> ms;
  ms.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  auto at = [&](double q) { return ms[std::min(ms.size() - 1, size_t(q * (ms.size() - 1) + 0.5))]; };
  return {at(0.5), at(0.9), ms.front()};
}

// Prevents the compiler from deleting a computation whose result is unused.
template <class T> inline void do_not_optimize(T const& v) { asm volatile("" : : "r,m"(v) : "memory"); }

struct Row { std::string label; double size; Timing t; double rate; double pct_peak; };

class Table {
 public:
  Table(std::string bench, std::string unit, std::string hardware = "unknown")
      : bench_(std::move(bench)), unit_(std::move(unit)), hw_(std::move(hardware)) {}

  void add(const std::string& label, double size, Timing t, double rate, double peak) {
    rows_.push_back({label, size, t, rate, peak > 0 ? 100.0 * rate / peak : 0.0});
  }

  void print() const {
    std::printf("\n| variant | problem size | median ms | p90 ms | %s | %% of peak |\n", unit_.c_str());
    std::printf("|---|---|---|---|---|---|\n");
    for (auto& r : rows_)
      std::printf("| %s | %.0f | %.3f | %.3f | %.2f | %.1f |\n", r.label.c_str(), r.size,
                  r.t.median_ms, r.t.p90_ms, r.rate, r.pct_peak);
  }

  // Writes the JSON contract described in course/common/README.md.
  void save_json(const std::string& path) const {
    std::ofstream f(path);
    f << "{\"bench\": \"" << bench_ << "\", \"hardware\": \"" << hw_ << "\", \"unit\": \"" << unit_
      << "\", \"rows\": [";
    for (size_t i = 0; i < rows_.size(); ++i) {
      auto& r = rows_[i];
      f << (i ? ", " : "") << "{\"label\": \"" << r.label << "\", \"size\": " << r.size
        << ", \"median_ms\": " << r.t.median_ms << ", \"p90_ms\": " << r.t.p90_ms
        << ", \"rate\": " << r.rate << ", \"pct_peak\": " << r.pct_peak << "}";
    }
    f << "]}\n";
  }

 private:
  std::string bench_, unit_, hw_;
  std::vector<Row> rows_;
};

}  // namespace s2s
