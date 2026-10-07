# P6.3: Paged KV block manager

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the block manager (pure Python, property-tested). ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for exercise 4 (paged decode attention kernel) |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P2.3 (PagedAttention as a user; D2), P5.8 exercise 4 (paged decode kernel), P6.2 |
| **You will build** | the KV block manager of #0 v1 (`platform/engine/v1/s2s_engine/block_manager.py`): block tables, ref counts, prefix reuse by chained block hashes, copy-on-write |

## Learning objectives

1. Map a token position to a physical cache slot through a block table, and say why fixed-size blocks kill external fragmentation.
2. Keep ref counts exact under sharing, forking and freeing — and test that with random operation sequences.
3. Reuse KV across requests by hashing **full** blocks, chained on their prefix, and keep freed-but-hashed blocks matchable until evicted.
4. Implement copy-on-write for forked sequences and hand the copies to the runner.
5. Connect the Python bookkeeping to the GPU: slot mapping for K/V writes, block tables for paged attention.

---

## 1. Slots and block tables

The KV pool is one big array of `num_blocks × block_size` token slots per layer. A sequence's block table maps its logical block i to a physical block id:

```
slot(pos) = block_table[pos // block_size] · block_size + pos % block_size
```

Every new token's K/V is written at `slot(pos)` (vLLM calls the per-token list the **slot mapping**); paged attention reads a sequence's K/V by walking its block table. Waste is at most `block_size − 1` slots per sequence (internal fragmentation), and there is no external fragmentation at all. The animation [`p2-paged-attention.html`](../../../animations/p2-paged-attention.html) (from P2.3) shows the mapping, sharing and CoW.

`NumpyPagedRunner` in `model_runner.py` is the whole idea in 30 lines: the v0 Llama forward with the contiguous cache replaced by slot lookups. `test_numpy_engine.py` checks it against v0 **token for token**, under chunking and preemption.

## 2. Ref counts and the free lists

| state | `ref` | where | meaning |
|---|---|---|---|
| in use | ≥ 1 | some block tables | `ref` = number of tables that contain it |
| cached-free | 0 | `cached_free` (LRU) | holds a full, hashed block; reusable by a prefix match until its slot is needed |
| free | 0 | `free` | no reusable content |

`_take()` prefers `free` and only then evicts the least recently freed cached block. `free_seq()` releases the **tail first**, so a sequence's prefix stays cached longest. The invariant `check_invariants()` checks after every random operation: `ref[b]` equals the number of block-table occurrences of b, and every block is in exactly one of {in use, free, cached-free}.

## 3. Prefix reuse by hashing full blocks

```
h_0 = H("",   tokens[0:bs])
h_i = H(h_{i-1}, tokens[i·bs:(i+1)·bs])
```

Chaining makes a block's hash depend on its whole prefix, which is exactly what its K/V depends on. Only **full** blocks are hashed (a partial block's content is still changing), and a match must leave **at least one prompt token uncomputed** so the model produces logits for the first output token. We use sha256, not Python's `hash()`, so hashes are stable across processes (vLLM's reason too; read how it handles hash choice at the pin, exercise 3).

> **Predict first.** Block size 16. Prompt A has 100 tokens and has finished. Prompt B shares A's first 70 tokens. How many tokens of B's prefill are skipped? (Answer: 64 — four full blocks; the fifth block differs at token 70.) What would a token-level radix cache skip? (P6.4.)

## 4. Copy-on-write

`fork(parent, child)` shares all of the parent's blocks (parallel sampling, beam search). Full blocks never change, so sharing them is free forever. The **last, partial** block will be written by both — so the first sequence to append into a shared partial block gets a private copy: `allocate()` takes a new block, queues `(src, dst)` in `copy_ops`, and swaps it into the table. The scheduler hands `copy_ops` to the runner, which copies the K/V **before** the forward pass.

## Walkthrough

```bash
uv run pytest platform/engine/v1/tests/test_block_manager.py platform/engine/v1/tests/test_numpy_engine.py -q
uv run pytest course/P6-engine-internals/P6.3-paged-kv-block-manager/exercises -q          # yours
uv run python course/P6-engine-internals/P6.3-paged-kv-block-manager/bench/utilization.py
```

## What you should see

- All property tests pass; `test_numpy_engine.py` prints nothing and passes (paged == contiguous, token for token).
- `utilization.py` prints, per block size, the fraction of allocated slots actually holding a token and the prefix-hit rate on a shared-system-prompt workload. Larger blocks → more internal waste and fewer prefix hits.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Block-table invariants under random operations](exercises/01-invariants.md) | T0 | `test_my_block_manager.py::test_random_*` |
| 2 | [Copy-on-write on fork](exercises/02-cow.md) | T0 | `test_my_block_manager.py::test_cow_*` |
| 3 | [Block-hash prefix reuse, and how vLLM does it](exercises/03-prefix-hash.md) | T0 | `test_my_block_manager.py::test_prefix_*` |
| 4 | [Paged decode attention kernel vs contiguous (from P5.8)](exercises/04-paged-kernel.md) | T2 | P5.8 `04-paged-decode` test |
| 5 | *(hard)* [Fragmentation and utilization under the #4 trace](exercises/05-fragmentation.md) | T0 | your table |

## Common mistakes

- Hashing a block before it is full, then matching it after more tokens were written.
- Matching the whole prompt from cache: no logits for the first output token.
- Returning a hashed block to `free` instead of `cached_free` — prefix hits silently stop working after the first request finishes.
- Executing CoW copies *after* the forward pass: the child reads its parent's newer token.

## Go deeper

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23).
- vLLM `vllm/v1/core/kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py` at `v0.31.0` (verify paths at the pin).
- nano-vllm `block_manager.py`. Modular handbook `pagedattention.md`.

**Next:** [P6.4 Prefix cache (radix tree)](../P6.4-prefix-cache-radix-tree/README.md).
