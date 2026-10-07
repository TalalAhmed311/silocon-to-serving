// Exercise 4 starter (hard): decode attention (q_len = 1) over a PAGED KV cache.
// q [B, H, 64]; k_cache, v_cache [num_blocks, block_size, Hkv, 64]; block_table [B, max_blocks] maps a sequence's
// logical block i to a physical block; seq_lens [B]; out [B, H, 64]. GQA: head h uses KV head h / (H / Hkv).
#pragma once
#include <cuda_fp16.h>

inline void paged_decode(const __half* q, const __half* k_cache, const __half* v_cache, const int* block_table,
                         const int* seq_lens, __half* out, int B, int H, int Hkv, int block_size, int max_blocks) {
  (void)q; (void)k_cache; (void)v_cache; (void)block_table; (void)seq_lens; (void)out; (void)B; (void)H; (void)Hkv;
  (void)block_size; (void)max_blocks;   // TODO
}
