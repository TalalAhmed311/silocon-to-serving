// d4/scan.cuh — inclusive float scan (P5.4): block scans (Hillis–Steele, warp-shuffle), the 3-phase
// scan-then-propagate for large arrays, and a single-pass DECOUPLED LOOK-BACK scan (Merrill & Garland 2016) — the
// idea behind CUB's DeviceScan.
#pragma once
#include "common.cuh"

namespace d4 {

constexpr int SCAN_T = 256, SCAN_IPT = 8, SCAN_TILE_N = SCAN_T * SCAN_IPT;   // 2048 elements per tile

// Inclusive warp scan with shuffles (Hillis–Steele inside a warp: log2(32) = 5 steps).
__device__ __forceinline__ float warp_incl_scan(float v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float u = __shfl_up_sync(FULL_MASK, v, o);
    if (lane >= o) v += u;
  }
  return v;
}

// Exclusive block scan of one value per thread; also returns the block total. blockDim.x == SCAN_T.
__device__ __forceinline__ float block_excl_scan(float v, float* total) {
  __shared__ float warp_tot[32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const float inc = warp_incl_scan(v);
  if (lane == 31) warp_tot[w] = inc;
  __syncthreads();
  if (w == 0) {
    const float x = lane < nw ? warp_tot[lane] : 0.f;
    const float xs = warp_incl_scan(x);
    if (lane < nw) warp_tot[lane] = xs;                 // inclusive scan of warp totals
  }
  __syncthreads();
  const float warp_prefix = w ? warp_tot[w - 1] : 0.f;
  *total = warp_tot[nw - 1];
  const float excl_in_warp = __shfl_up_sync(FULL_MASK, inc, 1);
  __syncthreads();                                       // warp_tot may be reused by the caller's next call
  return warp_prefix + (lane ? excl_in_warp : 0.f);
}

// Load a 2048-element tile (coalesced, via smem), scan it with `carry_in` added, store it. Returns the tile total.
__device__ __forceinline__ void tile_load(const float* in, long long base, int n, float* sm) {
  for (int k = threadIdx.x; k < SCAN_TILE_N; k += blockDim.x) sm[k] = base + k < n ? in[base + k] : 0.f;
  __syncthreads();
}

__device__ __forceinline__ float tile_scan_local(float* sm, float loc[SCAN_IPT]) {
  float run = 0.f;
#pragma unroll
  for (int k = 0; k < SCAN_IPT; ++k) { run += sm[threadIdx.x * SCAN_IPT + k]; loc[k] = run; }
  return run;
}

__device__ __forceinline__ void tile_store(float* out, long long base, int n, float* sm, const float loc[SCAN_IPT], float prefix) {
#pragma unroll
  for (int k = 0; k < SCAN_IPT; ++k) sm[threadIdx.x * SCAN_IPT + k] = loc[k] + prefix;
  __syncthreads();
  for (int k = threadIdx.x; k < SCAN_TILE_N; k += blockDim.x) if (base + k < n) out[base + k] = sm[k];
}

// ---- 3-phase: scan tiles + write totals → scan totals (recursively) → add carries -------------------------------
__global__ void scan_tiles_k(const float* in, float* out, float* totals, int n) {
  __shared__ float sm[SCAN_TILE_N];
  const long long base = (long long)blockIdx.x * SCAN_TILE_N;
  tile_load(in, base, n, sm);
  float loc[SCAN_IPT];
  const float run = tile_scan_local(sm, loc);
  float total;
  const float pre = block_excl_scan(run, &total);
  tile_store(out, base, n, sm, loc, pre);
  if (totals && threadIdx.x == 0) totals[blockIdx.x] = total;
}

__global__ void add_carry_k(float* out, const float* scanned_totals, int n) {
  if (blockIdx.x == 0) return;
  const float c = scanned_totals[blockIdx.x - 1];
  const long long base = (long long)blockIdx.x * SCAN_TILE_N;
  for (int k = threadIdx.x; k < SCAN_TILE_N; k += blockDim.x) if (base + k < n) out[base + k] += c;
}

inline void scan_3phase(const float* in, float* out, int n, cudaStream_t st = 0) {
  if (n <= 0) return;
  const int tiles = ceil_div(n, SCAN_TILE_N);
  if (tiles == 1) { scan_tiles_k<<<1, SCAN_T, 0, st>>>(in, out, nullptr, n); return; }
  float* totals = nullptr;
  cudaMallocAsync(&totals, size_t(tiles) * sizeof(float), st);
  scan_tiles_k<<<tiles, SCAN_T, 0, st>>>(in, out, totals, n);
  scan_3phase(totals, totals, tiles, st);
  add_carry_k<<<tiles, SCAN_T, 0, st>>>(out, totals, n);
  cudaFreeAsync(totals, st);
}

// ---- single pass with decoupled look-back -------------------------------------------------------------------------
// Each tile publishes (flag, value) packed in 64 bits: flag 0 = nothing yet, 1 = AGGREGATE (its own total), 2 = PREFIX
// (inclusive total of all tiles up to and including it). A tile's thread 0 walks backwards over predecessors, adding
// aggregates until it meets a PREFIX — so most tiles never wait for the whole chain. Tile ids come from an atomic
// counter (not blockIdx) so a tile only ever waits on tiles that have already STARTED: no deadlock.
__device__ __forceinline__ unsigned long long pack(unsigned flag, float v) {
  return (static_cast<unsigned long long>(flag) << 32) | __float_as_uint(v);
}

__global__ void scan_lookback_k(const float* in, float* out, int n, unsigned long long* status, unsigned* tile_counter) {
  __shared__ float sm[SCAN_TILE_N];
  __shared__ int tile_s;
  __shared__ float exclusive_s;
  if (threadIdx.x == 0) tile_s = int(atomicAdd(tile_counter, 1u));
  __syncthreads();
  const int tile = tile_s;
  const long long base = (long long)tile * SCAN_TILE_N;
  tile_load(in, base, n, sm);
  float loc[SCAN_IPT];
  const float run = tile_scan_local(sm, loc);
  float total;
  const float pre = block_excl_scan(run, &total);
  if (threadIdx.x == 0) {
    volatile unsigned long long* vs = status;
    if (tile == 0) {
      atomicExch(&status[0], pack(2, total));
      exclusive_s = 0.f;
    } else {
      atomicExch(&status[tile], pack(1, total));         // publish my aggregate early: successors can use it
      float acc = 0.f;
      for (int j = tile - 1; j >= 0;) {
        const unsigned long long s = vs[j];
        const unsigned flag = unsigned(s >> 32);
        if (flag == 0) continue;                          // predecessor hasn't published yet: spin
        acc += __uint_as_float(unsigned(s & 0xffffffffu));
        if (flag == 2) break;                             // inclusive prefix found: done
        --j;
      }
      atomicExch(&status[tile], pack(2, acc + total));
      exclusive_s = acc;
    }
  }
  __syncthreads();
  tile_store(out, base, n, sm, loc, pre + exclusive_s);
}

// status must hold ceil(n / 2048) uint64 and be zeroed with the counter before every call (done here).
inline void scan_lookback(const float* in, float* out, int n, unsigned long long* status, unsigned* counter, cudaStream_t st = 0) {
  if (n <= 0) return;
  const int tiles = ceil_div(n, SCAN_TILE_N);
  cudaMemsetAsync(status, 0, size_t(tiles) * sizeof(unsigned long long), st);
  cudaMemsetAsync(counter, 0, sizeof(unsigned), st);
  scan_lookback_k<<<tiles, SCAN_T, 0, st>>>(in, out, n, status, counter);
}

}  // namespace d4
