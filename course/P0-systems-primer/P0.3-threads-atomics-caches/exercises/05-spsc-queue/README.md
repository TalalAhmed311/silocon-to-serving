# Exercise 5 — Lock-free single-producer/single-consumer queue (hard)

The scheduler thread produces "blocks to free" and a cleanup thread consumes them. With exactly **one** producer and **one** consumer, a bounded ring buffer needs no lock. It needs two atomic indices with the right memory ordering.

```cpp
template <class T, size_t N>   // N is a power of two
class SpscQueue {
 public:
  bool push(const T& v);   // producer only; false if full
  bool pop(T& out);        // consumer only; false if empty
};
```

**Hints**

1. `head_` is written only by the consumer and `tail_` only by the producer. Each side reads the other's index.
2. Producer: write the slot, then `tail_.store(t + 1, release)`. Consumer: `tail_.load(acquire)`, then read the slot. This pair is what makes the slot's contents visible to the consumer.
3. Use free-running `size_t` counters and index with `& (N - 1)`. Full is `tail - head == N`, empty is `tail == head`.
4. Put `head_` and `tail_` on different cache lines (exercise 2).

**Test:** the producer pushes 0..2,000,000 while the consumer pops them. The test checks that nothing is lost and the order is strictly increasing. Run it with `-DS2S_TSAN=ON`: TSan must stay silent.
