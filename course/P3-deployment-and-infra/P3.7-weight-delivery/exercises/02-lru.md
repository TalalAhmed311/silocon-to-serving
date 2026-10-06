# Exercise 2: LRU NVMe cache with a size limit (T0)

The cache (`platform/weights/cache.py`) delegates the decision of *what to evict* to a pure function:

```python
plan_evictions(lru: dict[key, last_used_time], sizes: dict[key, bytes], pinned: set[key], need: int, max_bytes: int) -> list[key]
```

Implement it in `evict_impl.py`:

- evict least-recently-used first, and stop as soon as `used + need <= max_bytes`
- never evict a pinned key (a running replica is reading it)
- raise `CacheFull` if `need > max_bytes`, or if evicting every unpinned key still isn't enough. **Plan first, then act:** a failed plan must evict nothing

`test_lru.py` tests your function. `test_cache.py` tests the whole `WeightCache` against a `DirStore` (hits and misses, atomic install, verification failure, pins, persistence across restarts). It always uses the reference, so read it to see how your function is used.

**Then (design):** two pods on the same node call `get()` for the same new version at the same time. The in-process lock doesn't help across pods. What happens with the current code? Sketch a fix (a lock file with `fcntl.flock`, or a single prefetch DaemonSet as the only writer) and say which one you'd ship.
