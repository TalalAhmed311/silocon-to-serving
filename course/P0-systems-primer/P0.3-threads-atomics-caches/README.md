# P0.3 — Threads, atomics, caches

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). NUMA measurements are T3-optional (a multi-socket instance) |
| **Time** | ≈35 min reading + ≈7 h hands-on |
| **Prerequisites** | P0.1, P0.2 |
| **You will build** | a thread pool (used by P0.5) and **D2, the KV block allocator** (grows into PagedAttention in P6.3) |

## Learning objectives

1. Contrast processes and threads: what they share, what each costs to create, and when to use which.
2. Use `std::thread`, `std::mutex`, `std::condition_variable` and `std::atomic` correctly, with acquire/release ordering explained.
3. Detect and fix **false sharing**, and explain it with the cache-coherence protocol.
4. Build a fixed-size thread pool with `parallel_for`.
5. Explain NUMA and why "where the memory lives" matters on multi-socket servers.
6. Read `perf stat` counters and a flame graph.

## Why this matters

An inference server is a concurrency machine:

- an HTTP thread pool accepts requests
- a scheduler thread forms batches
- worker threads (or GPU streams) execute them
- a ref-counted block allocator hands KV-cache blocks to sequences that may share prefixes

D2 in this module is a direct precursor of vLLM's KV block pool. The thread pool parallelizes every matmul in your P0.5 engine.

---

## 1. Processes vs threads

| | Process | Thread |
|---|---|---|
| address space | its own (page tables from P0.2) | shared with the other threads of its process |
| creation | `fork`/`exec`: copy-on-write page tables, ~100s of µs | `pthread_create`: a stack + a kernel task, ~10s of µs |
| communication | pipes, sockets, shared memory | plain memory (and plain bugs) |
| crash isolation | yes | no: one segfault kills them all |

`examples/01_proc_vs_thread.c` measures both creation costs on your machine. Serving systems use **both**. vLLM runs separate processes for the API server and the engine core, partly so Python's GIL doesn't serialize them, and threads or async tasks inside each process.

## 2. Data races, mutexes and atomics

Two threads doing `counter++` on a shared `int` is a **data race**. `++` is a load, an add and a store, and two threads can interleave those steps and lose updates. Data races are undefined behavior in C++. Three fixes, from slowest to fastest:

1. **`std::mutex`**: only one thread at a time in the critical section. It is simple, but under contention every thread queues.
2. **`std::atomic<int>`**: `fetch_add` is a single indivisible read-modify-write (on x86, a `lock xadd`). There is no lock, but the cache line holding the counter still bounces between cores.
3. **Per-thread counters, summed at the end**: no sharing at all. When it's possible, it is always the fastest.

`examples/03_atomics.cpp` runs all three and prints a table.

### Memory ordering, the 5-minute version

Atomics have a second job: **ordering** other memory operations. The pattern you need most often is publishing data from one thread to another:

```cpp
// producer                                   // consumer
data = 42;                                    while (!ready.load(std::memory_order_acquire)) {}
ready.store(true, std::memory_order_release); assert(data == 42);   // guaranteed
```

A **release** store makes every write before it visible to any thread that **acquire**-loads the same atomic and sees the stored value. `memory_order_seq_cst`, the default, is stronger and simpler to reason about. Use it unless a profiler tells you otherwise. `relaxed` gives atomicity only, which is fine for statistics counters and wrong for flags that publish data.

## 3. Caches and coherence

A CPU core reads memory in **64-byte cache lines**, through private L1 and L2 caches and a shared L3. When two cores cache the same line, a **coherence protocol** keeps them consistent. In its simplest form, MESI, each line in each cache is **M**odified, **E**xclusive, **S**hared or **I**nvalid. A write requires the line to be in M or E, which means **invalidating every other core's copy**.

### False sharing

Now put two *independent* counters, one per thread, side by side in memory:

```cpp
struct { long a; long b; } counters;   // a and b share one 64-byte line
// thread 0: counters.a++ forever        thread 1: counters.b++ forever
```

The threads never touch each other's data, yet each write invalidates the other core's copy of the line. The line ping-pongs across the interconnect on every increment. This is **false sharing**. The fix is to give each counter its own line:

```cpp
struct alignas(64) Padded { long v; };   // or std::hardware_destructive_interference_size
Padded counters[2];
```

> **Predict first.** On your laptop, how much faster is the padded version with 2 threads? With 4? Write down a factor, then run `bench/false_sharing_bench`. (Expect a large factor: an L1 hit is a few cycles, while a coherence miss to another core is tens to a hundred-plus cycles. The exact ratio is your measurement.)

## 4. A thread pool

Creating a thread per task costs about 10 µs each time. A **thread pool** creates N workers once, and they pull tasks from a queue. For data-parallel loops, a `parallel_for(n, fn)` splits `[0, n)` into one contiguous chunk per worker. Contiguous chunks matter: each thread then streams its own region of memory and doesn't false-share output lines.

`examples/04_thread_pool.cpp` is ~80 lines:

- a `std::vector<std::thread>`
- a `std::deque<std::function<void()>>` guarded by a mutex
- a `condition_variable` that wakes idle workers
- `parallel_for`, which enqueues chunks and blocks on a countdown until all of them finish

## 5. NUMA

On a multi-socket server, each socket has its own memory controllers. Memory attached to the *other* socket is reachable, but slower and with less bandwidth: **Non-Uniform Memory Access**. Linux allocates a page on the node of the thread that **first touches** it (P0.2's demand paging again). So the thread that initializes a buffer decides where it lives. `numactl --hardware` shows the topology, and `numactl --cpunodebind=0 --membind=0 ./app` pins a run to one node. GPU servers have the same issue: a GPU hangs off one socket's PCIe root complex, so pinned host buffers for that GPU should live on that socket's memory. That is why `nvidia-smi topo -m` matters (P4).

## 6. `perf` in one screen

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses ./build/examples/02_false_sharing
perf record -g ./build/examples/02_false_sharing && perf report         # where time goes
perf c2c record ./build/examples/02_false_sharing && perf c2c report    # finds false sharing directly
```

`examples/05_perf_walkthrough.md` explains how to read each counter. Flame graphs (Brendan Gregg's `flamegraph.pl`) turn `perf record -g` output into a picture of the call stacks. macOS: use Instruments → "CPU Counters" and "Time Profiler".

---

## Walkthrough

```bash
cd course/P0-systems-primer/P0.3-threads-atomics-caches
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release && cmake --build build/examples -j
./build/examples/01_proc_vs_thread
./build/examples/02_false_sharing 4
./build/examples/03_atomics 4
./build/examples/04_thread_pool
```

## What you should see

All timings are machine-dependent. **TODO(run): paste your tables.** Expected shape:

- `01`: thread creation is an order of magnitude cheaper than `fork` + wait.
- `02`: the padded counters are several times faster than the packed ones, and the gap grows with the thread count.
- `03`: per-thread < atomic < mutex in time, with the mutex degrading most as the thread count grows.
- `04`: `parallel_for` sum = serial sum exactly (integers), with a speedup close to the number of physical cores for a compute-bound body.

## Exercises

```bash
cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex --output-on-failure
cmake -S exercises -B build/ex-sol -DS2S_USE_SOLUTIONS=ON && cmake --build build/ex-sol -j && ctest --test-dir build/ex-sol
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Fix the data race](exercises/01-fix-the-race/README.md) | easy | exact count; also runs under ThreadSanitizer |
| 2 | [Pad the counters](exercises/02-pad-counters/README.md) | easy | `alignof`/`sizeof` checks + bench row |
| 3 | [`ThreadPool::parallel_for`](exercises/03-thread-pool/README.md) | medium | parallel sums vs serial, n < workers, 1000 back-to-back calls, exceptions |
| 4 | [**D2: KV block allocator**](exercises/04-kv-block-allocator/README.md) | medium | invariants under a seeded multi-threaded stress test |
| 5 | [Lock-free SPSC block queue](exercises/05-spsc-queue/README.md) | hard | ordering + no loss under TSan |

## Benchmark

```bash
cmake -S bench -B build/bench -DCMAKE_BUILD_TYPE=Release && cmake --build build/bench -j
./build/bench/false_sharing_bench     # | threads | packed ns/op | padded ns/op | speedup |
./build/bench/allocator_bench         # | threads | allocs/s (mutex) | ... (your D2) |
```

Both write `results/*.json`.

## Common mistakes

- **`volatile` as a synchronization tool.** It isn't one. Use `std::atomic`.
- **Waiting on a condition variable without a predicate.** Spurious wakeups happen. Always write `cv.wait(lock, [&]{ return ready; })`.
- **Holding a lock while calling user code**, such as a task body or a callback. That invites deadlock. Pop the task, unlock, then run it.
- **Benchmarking `std::atomic` on one thread.** Uncontended atomics are cheap. The cost appears only under contention.
- **Assuming `std::hardware_destructive_interference_size` exists.** Some standard libraries don't define it. The course code falls back to 64. Apple M-series chips use 128-byte lines for some purposes, so check your platform.

## Go deeper

- OSTEP Part II, Concurrency: *Threads*, *Locks*, *Condition Variables*, *Common Concurrency Problems*.
- *C++ Concurrency in Action*, 2nd ed.: ch. 2–4 (threads, sharing, synchronization), ch. 5 (the memory model), ch. 9 (thread pools).
- CS:APP ch. 6.4–6.6 (cache organization), ch. 12 (concurrent programming).
- Brendan Gregg, *perf Examples* and *Flame Graphs*.

## Go down when…

Bottom of the CPU path. **Next:** [P0.4 SIMD and roofline](../P0.4-simd-and-roofline/README.md). **Comes back in:** P2.3 (prefix caching = shared ref-counted blocks), P6.3 (paged KV block manager).

## Animation

[`animations/p0-false-sharing.html`](../../../animations/p0-false-sharing.html) shows two cores writing two counters on one cache line. Watch the MESI states and the invalidation traffic on every write, then switch to the padded layout.
