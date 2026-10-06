// LeetGPU #36 Radix Sort — Lane B L3 solution. Sorts N unsigned 32-bit keys ascending (LSD, 8-bit digits, 4 passes).
// Per pass: (1) each 2048-key tile is stably sorted by the digit in shared memory (8 one-bit splits) and writes its
// 256-bin digit histogram column-major; (2) an exclusive scan of that table gives every (digit, tile) its global
// offset; (3) each tile scatters its locally-sorted keys. Stable, so LSD works. Our signature; check the statement.
#include <cuda_runtime.h>

// ---- generic tile scan (identical in every L3 scan solution, so each kernel.cu stays self-contained) ----------------
// inclusive_scan(in, out, n, op, identity): out[i] = in[0] op in[1] op ... op in[i]. `op` must be associative (not
// necessarily commutative); it is always applied as op(earlier, later). Three phases: scan each 2048-element tile
// and emit its total → scan the totals (recursively) → fold each tile's carry-in into it. In-place (in == out) is OK.
constexpr int SCAN_THREADS = 256, SCAN_ITEMS = 8, SCAN_TILE = SCAN_THREADS * SCAN_ITEMS;

// Exclusive scan of one value per thread (Hillis–Steele over 256 slots in shared memory; works for any struct T).
template <class T, class Op>
__device__ T block_exclusive_scan(T v, Op op, T identity, T* block_total) {
  __shared__ T s[SCAN_THREADS];
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = 1; o < SCAN_THREADS; o <<= 1) {
    T x = identity;
    if (t >= o) x = s[t - o];
    __syncthreads();
    if (t >= o) s[t] = op(x, s[t]);
    __syncthreads();
  }
  const T excl = t ? s[t - 1] : identity;
  *block_total = s[SCAN_THREADS - 1];
  __syncthreads();
  return excl;
}

template <class T, class Op>
__global__ void scan_tiles(const T* in, T* out, T* tile_totals, int n, Op op, T identity) {
  __shared__ T tile[SCAN_TILE];
  const long long base = (long long)blockIdx.x * SCAN_TILE;
  for (int i = threadIdx.x; i < SCAN_TILE; i += SCAN_THREADS)      // coalesced load into shared memory
    tile[i] = (base + i < n) ? in[base + i] : identity;
  __syncthreads();
  T* mine = tile + threadIdx.x * SCAN_ITEMS;                          // each thread scans 8 consecutive items
  T run = mine[0];
  T loc[SCAN_ITEMS];
  loc[0] = run;
#pragma unroll
  for (int k = 1; k < SCAN_ITEMS; ++k) { run = op(run, mine[k]); loc[k] = run; }
  T total;
  const T prefix = block_exclusive_scan(run, op, identity, &total);
#pragma unroll
  for (int k = 0; k < SCAN_ITEMS; ++k) mine[k] = op(prefix, loc[k]);
  __syncthreads();
  for (int i = threadIdx.x; i < SCAN_TILE; i += SCAN_THREADS)
    if (base + i < n) out[base + i] = tile[i];
  if (tile_totals && threadIdx.x == 0) tile_totals[blockIdx.x] = total;
}

template <class T, class Op>
__global__ void add_carry(T* out, const T* scanned_totals, int n, Op op) {
  if (blockIdx.x == 0) return;
  const T carry = scanned_totals[blockIdx.x - 1];
  const long long base = (long long)blockIdx.x * SCAN_TILE;
  for (int i = threadIdx.x; i < SCAN_TILE; i += SCAN_THREADS)
    if (base + i < n) out[base + i] = op(carry, out[base + i]);
}

template <class T, class Op>
void inclusive_scan(const T* in, T* out, int n, Op op, T identity) {
  if (n <= 0) return;
  const int tiles = (n + SCAN_TILE - 1) / SCAN_TILE;
  if (tiles == 1) { scan_tiles<<<1, SCAN_THREADS>>>(in, out, (T*)nullptr, n, op, identity); return; }
  T* totals = nullptr;
  cudaMalloc(&totals, size_t(tiles) * sizeof(T));                    // production: one preallocated workspace
  scan_tiles<<<tiles, SCAN_THREADS>>>(in, out, totals, n, op, identity);
  inclusive_scan(totals, totals, tiles, op, identity);              // 2048× fewer elements per level
  add_carry<<<tiles, SCAN_THREADS>>>(out, totals, n, op);
  cudaFree(totals);
}
// ---------------------------------------------------------------------------------------------------------------------

constexpr int RADIX_BITS = 8, BINS = 1 << RADIX_BITS;

struct UAdd {
  __host__ __device__ unsigned operator()(unsigned a, unsigned b) const { return a + b; }
};

// Stable sort of a shared-memory tile by `bits` bits starting at `shift`: one stable 0/1 split per bit.
__device__ void local_digit_sort(unsigned* keys, unsigned* tmp, int shift) {
  const int t = threadIdx.x;
  for (int b = shift; b < shift + RADIX_BITS; ++b) {
    unsigned zeros = 0;                                   // zeros among my 8 consecutive keys
#pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) zeros += ((keys[t * SCAN_ITEMS + k] >> b) & 1u) ^ 1u;
    unsigned total_zeros;
    unsigned before = block_exclusive_scan(zeros, UAdd{}, 0u, &total_zeros);
#pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
      const int p = t * SCAN_ITEMS + k;
      const unsigned key = keys[p];
      if (((key >> b) & 1u) == 0) tmp[before++] = key;                      // zeros keep their relative order
      else tmp[total_zeros + (unsigned(p) - before)] = key;                 // ones: after all zeros, in order
    }
    __syncthreads();
    for (int i = t; i < SCAN_TILE; i += SCAN_THREADS) keys[i] = tmp[i];
    __syncthreads();
  }
}

__global__ void tile_sort_and_count(const unsigned* __restrict__ in, unsigned* __restrict__ sorted, unsigned* __restrict__ hist,
                                    int n, int shift, int tiles) {
  __shared__ unsigned keys[SCAN_TILE], tmp[SCAN_TILE], h[BINS];
  const long long base = (long long)blockIdx.x * SCAN_TILE;
  for (int i = threadIdx.x; i < SCAN_TILE; i += SCAN_THREADS) keys[i] = base + i < n ? in[base + i] : 0xFFFFFFFFu;  // pad sorts last
  for (int i = threadIdx.x; i < BINS; i += SCAN_THREADS) h[i] = 0;
  __syncthreads();
  local_digit_sort(keys, tmp, shift);
  const int valid = (n - base) < SCAN_TILE ? int(n - base) : SCAN_TILE;
  for (int i = threadIdx.x; i < valid; i += SCAN_THREADS) atomicAdd(&h[(keys[i] >> shift) & (BINS - 1)], 1u);
  __syncthreads();
  for (int i = threadIdx.x; i < SCAN_TILE; i += SCAN_THREADS) sorted[base + i] = keys[i];   // sorted buffer padded to tiles*TILE
  for (int d = threadIdx.x; d < BINS; d += SCAN_THREADS) hist[d * tiles + blockIdx.x] = h[d];   // digit-major
}

// offsets = exclusive scan of hist (digit-major): all digit-0 keys of all tiles first, then digit 1, ...
__global__ void scatter(const unsigned* __restrict__ sorted, const unsigned* __restrict__ incl, const unsigned* __restrict__ hist,
                        unsigned* __restrict__ out, int n, int shift, int tiles) {
  __shared__ unsigned start[BINS], goff[BINS];
  const long long base = (long long)blockIdx.x * SCAN_TILE;
  for (int d = threadIdx.x; d < BINS; d += SCAN_THREADS) {
    const int idx = d * tiles + blockIdx.x;
    goff[d] = incl[idx] - hist[idx];          // exclusive offset of (digit d, this tile)
  }
  // local start of each digit inside the sorted tile = exclusive scan of this tile's histogram (256 entries, serial ok)
  if (threadIdx.x == 0) {
    unsigned run = 0;
    for (int d = 0; d < BINS; ++d) { start[d] = run; run += hist[d * tiles + blockIdx.x]; }
  }
  __syncthreads();
  const int valid = (n - base) < SCAN_TILE ? int(n - base) : SCAN_TILE;
  for (int i = threadIdx.x; i < valid; i += SCAN_THREADS) {
    const unsigned key = sorted[base + i];
    const unsigned d = (key >> shift) & (BINS - 1);
    out[goff[d] + (unsigned(i) - start[d])] = key;
  }
}

void solve(unsigned* data, int N) {
  if (N <= 1) return;
  const int tiles = (N + SCAN_TILE - 1) / SCAN_TILE;
  unsigned *sorted, *hist, *incl, *other;
  cudaMalloc(&sorted, size_t(tiles) * SCAN_TILE * sizeof(unsigned));
  cudaMalloc(&hist, size_t(tiles) * BINS * sizeof(unsigned));
  cudaMalloc(&incl, size_t(tiles) * BINS * sizeof(unsigned));
  cudaMalloc(&other, size_t(N) * sizeof(unsigned));
  unsigned *src = data, *dst = other;
  for (int shift = 0; shift < 32; shift += RADIX_BITS) {       // 4 passes: data → other → data → other → data
    tile_sort_and_count<<<tiles, SCAN_THREADS>>>(src, sorted, hist, N, shift, tiles);
    inclusive_scan(hist, incl, tiles * BINS, UAdd{}, 0u);
    scatter<<<tiles, SCAN_THREADS>>>(sorted, incl, hist, dst, N, shift, tiles);
    unsigned* t = src; src = dst; dst = t;
  }
  cudaFree(sorted); cudaFree(hist); cudaFree(incl); cudaFree(other);
}
