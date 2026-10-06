// Exercise 3 solution (the same algorithm as examples/thread_pool.hpp).
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n = 4) { for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); }); }
  ~ThreadPool() {
    { std::lock_guard g(mu_); stop_ = true; }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  unsigned size() const { return unsigned(workers_.size()); }
  void submit(std::function<void()> t) { { std::lock_guard g(mu_); q_.push_back(std::move(t)); } cv_.notify_one(); }

  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    const int64_t chunks = std::min<int64_t>(n, size()), step = (n + chunks - 1) / chunks;
    std::mutex m;
    std::condition_variable cv;
    int64_t left = 0;
    std::exception_ptr err;
    for (int64_t b = 0; b < n; b += step) ++left;   // exact chunk count (the last may be short)
    for (int64_t b = 0; b < n; b += step) {
      const int64_t e = std::min(n, b + step);
      submit([&, b, e] {
        std::exception_ptr local;
        try { fn(b, e); } catch (...) { local = std::current_exception(); }
        std::lock_guard g(m);
        if (local && !err) err = local;
        if (--left == 0) cv.notify_one();
      });
    }
    std::unique_lock lk(m);
    cv.wait(lk, [&] { return left == 0; });
    if (err) std::rethrow_exception(err);
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> t;
      { std::unique_lock lk(mu_); cv_.wait(lk, [&] { return stop_ || !q_.empty(); }); if (stop_ && q_.empty()) return;
        t = std::move(q_.front()); q_.pop_front(); }
      t();
    }
  }
  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> q_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};
