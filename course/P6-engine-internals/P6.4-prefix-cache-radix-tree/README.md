# P6.4: Prefix cache with a radix tree

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) everything |
| **Time** | ≈30 min reading + ≈9 h hands-on |
| **Prerequisites** | P6.3 (block-hash prefix reuse) |
| **You will build** | a RadixAttention-style prefix cache (`platform/engine/v1/s2s_engine/radix_cache.py`): insert, longest-prefix match with edge splits, ref-count locks, LRU eviction of unlocked leaves — tested against a naive trie |

## Learning objectives

1. Represent every cached token sequence in one compressed trie whose edges carry tokens *and* their KV slots.
2. Implement longest-prefix match that splits an edge on a partial match, and insert that shares existing prefixes.
3. Protect in-use KV with ref counts along the root path, and evict least-recently-used unlocked leaves.
4. Compare token-granular radix matching with full-block hash matching (P6.3) on real workload shapes.
5. Explain cache-aware scheduling and measure what it buys.

---

## 1. The structure

The animation [`p6-radix-tree.html`](../../../animations/p6-radix-tree.html) walks through every operation below.

```
root ─"SYSTEM"─┬─"hello"           Each edge: a run of tokens + the KV slot of each token.
               ├─"hi"              Children are keyed by their edge's first token.
               └─"bye"             A path from the root spells a cached sequence.
```

| operation | what it does | cost |
|---|---|---|
| `match_prefix(tokens)` | walk edges while tokens agree; on a partial edge match, **split** the edge so the matched part is a node | O(matched length) |
| `insert(tokens, slots)` | match, then hang the unmatched suffix as a new leaf; returns how much was already cached (the caller frees its duplicate slots) | O(length) |
| `lock(node)` / `unlock(node)` | ref count +1 / −1 on every node from `node` to the root | O(depth) |
| `evict(n)` | pop LRU **unlocked leaves** until ≥ n tokens freed; a parent that loses its last child joins the heap | O(k log k) |

SGLang's `RadixCache` (`python/sglang/srt/mem_cache/radix_cache.py` at `v0.5.21`, verify) adds page-size alignment, a separate token-to-KV pool, and hooks for hierarchical caching, but the four operations are the same.

## 2. Radix vs block hashing

| | block hash (P6.3, vLLM) | radix tree (SGLang) |
|---|---|---|
| match granularity | full blocks only | any token |
| lookup | one hash-table probe per block | one trie walk |
| shared-prefix storage | implicit (same hash → same block) | explicit tree |
| eviction unit | a block (LRU over freed blocks) | a leaf's token run (LRU over unlocked leaves) |
| cross-process stability | needs a stable hash (sha256) | tree lives in one process |

On a multi-turn chat, each turn's prompt is the previous prompt + answer + new message: both schemes hit almost everything, the radix tree also catches the last partial block. On few-shot prompts that share a long header and differ at a random token offset, the radix tree's advantage is up to `block_size − 1` tokens per request — small for long prompts, large for short ones. `bench/hit_rate.py` measures both.

> **Predict first.** Block size 16; 1,000 requests share a 50-token system prompt, then diverge. Hit tokens per request: block hash? radix? (48 vs 50.) Now the shared prefix is 500 tokens: is the difference worth a second data structure?

## 3. Cache-aware scheduling

If the waiting queue holds requests that share prefixes, serving the ones with the **longest cached prefix first** (SGLang's LPM policy) increases hits: their prefix is locked and reused before eviction can take it. The risk is unfairness (requests with no shared prefix wait), so combine it with P6.2's aging. Exercise 4 measures both effects on a trace.

## Walkthrough

```bash
uv run pytest platform/engine/v1/tests/test_radix_cache.py -q
uv run pytest course/P6-engine-internals/P6.4-prefix-cache-radix-tree/exercises -q          # yours
uv run python course/P6-engine-internals/P6.4-prefix-cache-radix-tree/bench/hit_rate.py
```

## What you should see

`hit_rate.py` prints, for a multi-turn chat trace and a few-shot trace, the fraction of prompt tokens served from cache by block hashing (block size 16) and by the radix tree. The radix tree is never lower; the gap is largest on the few-shot trace with short prompts.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Insert / match / split vs a naive trie](exercises/01-radix.md) | T0 | `test_my_radix.py::test_match_*`, `test_split_*` |
| 2 | [LRU eviction that respects ref counts](exercises/02-evict.md) | T0 | `test_my_radix.py::test_evict*` |
| 3 | [Hit rate on multi-turn chat and few-shot traces](exercises/03-hit-rate.md) | T0 | your table |
| 4 | *(hard)* [Cache-aware (longest-prefix-first) scheduling](exercises/04-cache-aware.md) | T0 | your table |

## Common mistakes

- Splitting an edge but forgetting to re-key the parent's child map (the first token of the upper node equals the old first token — the child map entry must point to the new node).
- Evicting an internal node: its children's KV positions depend on its tokens.
- Locking only the matched node, not its ancestors.
- Inserting and keeping your own duplicate slots for the already-cached prefix: a KV leak.

## Go deeper

- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (arXiv 2312.07104), §3 RadixAttention.
- SGLang `radix_cache.py` and `schedule_policy.py` at the pin. Modular handbook `prefix-caching.md`.

**Next:** [P6.5 Spec-decode verification and GPU sampling](../P6.5-spec-decode-and-gpu-sampling/README.md).
