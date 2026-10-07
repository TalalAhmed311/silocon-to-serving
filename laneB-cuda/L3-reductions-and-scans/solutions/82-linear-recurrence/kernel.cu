// LeetGPU #82 Linear Recurrence — Lane B L3 solution. x[t] = a[t] * x[t-1] + b[t], with x[-1] = 0 (so x[0] = b[0]).
// Our signature; check the statement (initial value, coefficient layout).
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

// An affine map x -> a*x + b. Applying f then g: x -> g.a*(f.a*x + f.b) + g.b = (g.a*f.a)*x + (g.a*f.b + g.b).
struct Affine {
  float a, b;
};

struct Compose {  // op(earlier, later) = "earlier, then later"
  __host__ __device__ Affine operator()(Affine f, Affine g) const { return {g.a * f.a, fmaf(g.a, f.b, g.b)}; }
};

__global__ void pack(const float* __restrict__ a, const float* __restrict__ b, Affine* __restrict__ p, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = {a[i], b[i]};
}

// The prefix composition maps x[-1] to x[t]; with x[-1] = 0, x[t] is just its offset b.
__global__ void unpack(const Affine* __restrict__ p, float* __restrict__ x, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = p[i].b;
}

void solve(const float* a, const float* b, float* x, int N) {
  Affine* p = nullptr;
  cudaMalloc(&p, size_t(N) * sizeof(Affine));
  const int blocks = (N + 255) / 256 < 4096 ? (N + 255) / 256 : 4096;
  pack<<<blocks, 256>>>(a, b, p, N);
  inclusive_scan(p, p, N, Compose{}, Affine{1.f, 0.f});   // identity map
  unpack<<<blocks, 256>>>(p, x, N);
  cudaFree(p);
}
