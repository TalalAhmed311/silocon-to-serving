# Exercise 2: LRU eviction that respects locks (T0)

Implement `lock`, `unlock` and `evict`. Only leaves with `ref == 0` are candidates, least-recent `last_access` first; when a leaf is removed and its parent becomes an unlocked leaf, the parent becomes a candidate too.

Tests: `test_eviction_is_lru_and_respects_locks` and `test_evicting_children_exposes_parent`.

**Then:** the engine must free the KV slots `evict` returns. In #0 v1 the block manager owns KV memory; sketch (in prose or code) how you'd make the radix cache the owner instead, at block granularity: what does a node's `value` hold, and what happens to a partial last block?
