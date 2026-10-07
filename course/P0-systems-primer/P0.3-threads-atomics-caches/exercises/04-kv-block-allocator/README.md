# Exercise 4 — D2: a KV block allocator (medium)

A paged KV cache, such as vLLM's PagedAttention, splits GPU memory into fixed-size **blocks**. Each block holds the K/V vectors of a few tokens (for example 16). A sequence owns a list of blocks. When two sequences share a prompt prefix, they **share** the prefix blocks and a ref count tracks the owners. A shared block that one owner wants to modify must first be **copied on write**.

You'll build the CPU-side bookkeeping now. It manages integer block IDs; the memory comes later. In P6.3 this same class manages real KV memory.

```cpp
class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks);
  std::optional<int> allocate();                     // a free block with refcount 1, or nullopt if none left
  std::vector<int> allocate_many(int n);             // all-or-nothing: n blocks, or an empty vector
  void share(int block);                             // ++refcount (another sequence now uses it)
  void free(int block);                              // --refcount; back to the free list at 0
  int copy_on_write(int block);                      // refcount 1: return the same id; else allocate a new one,
                                                     //   drop one ref on the old, return the new (caller copies data)
  int refcount(int block) const;
  int num_free() const;
  int num_blocks() const;
};
```

**Requirements**

- All operations are thread-safe. One `std::mutex` is fine, and the bench measures how fine it is.
- `free` or `share` of a block with refcount 0, or an id out of range, throws `std::logic_error`. These are the double-free and use-after-free bugs of a KV cache.
- `allocate` is O(1): use a free *list* (a stack of ids), never a scan.
- Hand out recently freed blocks first (LIFO). Their memory is more likely to still be in cache, which is the same reason vLLM keeps a free-block queue.

**Hints**

1. Use `std::vector<int> refcnt` and `std::vector<int> free_stack`.
2. `allocate_many` checks `free_stack.size() >= n` *under the same lock* before popping anything.
3. `copy_on_write` = `allocate` + `free(old)`, done atomically under the lock.

**Test:** unit tests for each method, then a seeded stress test. 8 threads do 20,000 random alloc/share/free/cow operations each. Each thread holds its own references, and the invariants checked at the end are:

- `num_free + #blocks with refcount > 0 == num_blocks`
- every refcount equals the number of references the threads hold
- everything is free after the threads release all references
