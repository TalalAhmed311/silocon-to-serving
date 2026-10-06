# Exercise 3 — `ThreadPool::parallel_for` (medium)

`impl.hpp` contains a working `ThreadPool` with `submit()`, but `parallel_for` is missing. Implement:

```cpp
// Split [0, n) into at most size() contiguous chunks, run fn(begin, end) for each on the pool, block until all
// finish. If any chunk throws, rethrow the first exception after all chunks have finished.
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);
```

**Hints**

1. Use a countdown of outstanding chunks, protected by a mutex and signalled with a condition variable. `std::latch` (C++20) also works.
2. Capture `b` and `e` by value in each task. Capture the countdown by reference: it outlives the tasks because you wait for them.
3. Catch inside the task, store a `std::exception_ptr`, and rethrow it after the wait.

**Test:** a parallel sum vs serial for n ∈ {0, 1, 3, 1000, 10⁶}; n smaller than the worker count; 1000 back-to-back calls (catches lost wake-ups); an exception propagates and the pool stays usable.

The reference version is `examples/thread_pool.hpp`, which other modules use. Try not to peek.
