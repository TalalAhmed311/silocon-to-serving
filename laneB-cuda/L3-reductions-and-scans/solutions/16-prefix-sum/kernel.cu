// LeetGPU #16 Prefix Sum — Lane B L3 solution. output[i] = input[0] + ... + input[i] (inclusive). Check the statement
// for inclusive vs exclusive; exclusive is output[i] = inclusive[i] - input[i] for ints, or a shifted write.
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

struct Add {
  __host__ __device__ float operator()(float a, float b) const { return a + b; }
};

void solve(const float* input, float* output, int N) { inclusive_scan(input, output, N, Add{}, 0.f); }
