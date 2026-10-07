#pragma once
#include <d4/attention.cuh>
inline void paged_decode(const __half* q, const __half* k_cache, const __half* v_cache, const int* block_table,
                         const int* seq_lens, __half* out, int B, int H, int Hkv, int block_size, int max_blocks) {
  d4::paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens, out, B, H, Hkv, block_size, max_blocks);
}
