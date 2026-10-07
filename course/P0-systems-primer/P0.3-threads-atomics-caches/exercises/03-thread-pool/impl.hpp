// Exercise 3 starter: implement parallel_for.
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
    // TODO
    (void)n; (void)fn;
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
