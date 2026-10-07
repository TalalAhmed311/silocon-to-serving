// thread_pool.hpp — a fixed-size thread pool with submit() and parallel_for().
// Used by: P0.3 example 04, P0.4 D1 (multithreaded SGEMM), P0.5 engine (#0 v0).
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

namespace s2s {

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n = std::max(1u, std::thread::hardware_concurrency())) {
    for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
  }
  ~ThreadPool() {
    { std::lock_guard g(mu_); stop_ = true; }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return unsigned(workers_.size()); }

  void submit(std::function<void()> task) {
    { std::lock_guard g(mu_); q_.push_back(std::move(task)); }
    cv_.notify_one();
  }

  // Splits [0, n) into contiguous chunks (one per worker), runs fn(begin, end) on each, and waits.
  // Contiguous chunks: each worker streams its own memory region and writes its own output lines.
  // The first exception thrown by any chunk is rethrown here after all chunks finish.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    const int64_t chunks = std::min<int64_t>(n, size());
    const int64_t step = (n + chunks - 1) / chunks;
    std::mutex done_mu;
    std::condition_variable done_cv;
    int64_t remaining = chunks;
    std::exception_ptr err;
    for (int64_t c = 0; c < chunks; ++c) {
      const int64_t b = c * step, e = std::min(n, b + step);
      submit([&, b, e] {
        try { fn(b, e); } catch (...) { std::lock_guard g(done_mu); if (!err) err = std::current_exception(); }
        std::lock_guard g(done_mu);
        if (--remaining == 0) done_cv.notify_one();
      });
    }
    std::unique_lock lk(done_mu);
    done_cv.wait(lk, [&] { return remaining == 0; });  // predicate: immune to spurious wakeups
    if (err) std::rethrow_exception(err);
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        task = std::move(q_.front());
        q_.pop_front();
      }
      task();  // run outside the lock: a long task must not block other workers from dequeuing
    }
  }
  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> q_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace s2s
