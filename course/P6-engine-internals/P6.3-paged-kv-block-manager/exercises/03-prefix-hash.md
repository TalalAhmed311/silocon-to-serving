# Exercise 3: block-hash prefix reuse (T0)

Implement `commit` (hash newly full blocks with `chain_hash`, register them in `hash_of`/`block_of`) and `match_prefix` (walk the prompt's full blocks, stop at the first miss, leave ≥ 1 token uncomputed, take a reference on each hit — pulling it out of `cached_free` if its ref was 0). Freed hashed blocks go to `cached_free`; `_take` evicts from it LRU only when `free` is empty, unregistering the hash.

Tests: `test_prefix_reuse_full_blocks_only_and_leaves_one_token`, `test_prefix_cached_blocks_evicted_lru_when_needed`, and the engine-level `platform/engine/v1/tests/test_scheduler.py::test_prefix_cache_saves_compute`.

**Then (reading):** in vLLM at `v0.31.0`, find how a block hash is computed (`vllm/v1/core/kv_cache_utils.py`; verify the path). What goes into it besides the token ids (think: LoRA, multimodal inputs, a cache salt)? Which hash function is used by default, and what option changes it? Write three sentences comparing it with ours.
