// LeetGPU #51 Max Subarray Sum — Lane B L3 solution: max over non-empty contiguous subarrays of sum (Kadane),
// computed as an ORDERED parallel reduction of 4-tuples. Our signature/semantics; check the statement (some variants
// fix a window length — then it is a sliding-window sum: prefix sums + max-reduce).
#include <cuda_runtime.h>

struct Seg {            // summary of a contiguous segment
  long long total;      // sum of all elements
  long long pre;        // best prefix sum (non-empty)
  long long suf;        // best suffix sum (non-empty)
  long long best;       // best subarray sum (non-empty)
};

__host__ __device__ inline long long mx(long long a, long long b) { return a > b ? a : b; }

// Associative, NOT commutative: combine(left, right) for adjacent segments, left first.
__host__ __device__ inline Seg combine(const Seg& l, const Seg& r) {
  return {l.total + r.total, mx(l.pre, l.total + r.pre), mx(r.suf, r.total + l.suf), mx(mx(l.best, r.best), l.suf + r.pre)};
}

__host__ __device__ inline Seg leaf(long long v) { return {v, v, v, v}; }

constexpr int THREADS = 256;

// Each block owns one contiguous chunk; each thread a contiguous sub-chunk (serial combine), then an in-order tree.
__global__ void seg_partials(const int* __restrict__ in, Seg* __restrict__ part, int N, int chunk) {
  __shared__ Seg s[THREADS];
  __shared__ bool hv[THREADS];
  const long long b0 = (long long)blockIdx.x * chunk;
  const long long end = (b0 + chunk < N) ? b0 + chunk : N;
  const int per = (chunk + THREADS - 1) / THREADS;
  const long long t0 = b0 + (long long)threadIdx.x * per;
  const long long t1 = (t0 + per < end) ? t0 + per : end;
  bool have = false;
  Seg acc{};
  for (long long i = t0; i < t1; ++i) {             // contiguous per thread: not coalesced, but each thread
    const Seg x = leaf(in[i]);                       // streams its own run; see README for the staged load
    acc = have ? combine(acc, x) : x;
    have = true;
  }
  s[threadIdx.x] = acc;
  hv[threadIdx.x] = have;
  __syncthreads();
  for (int stride = 1; stride < THREADS; stride <<= 1) {   // in-order pairwise tree: (0,1),(2,3)… then (0,2)…
    const int i = threadIdx.x * 2 * stride;
    if (i + stride < THREADS) {
      if (hv[i] && hv[i + stride]) s[i] = combine(s[i], s[i + stride]);
      else if (hv[i + stride]) { s[i] = s[i + stride]; hv[i] = true; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}

__global__ void seg_final(const Seg* __restrict__ part, int n, long long* __restrict__ out) {
  Seg acc = part[0];
  for (int i = 1; i < n; ++i) acc = combine(acc, part[i]);   // a few hundred partials: one thread is fine
  *out = acc.best;
}

void solve(const int* input, long long* output, int N) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 4;
  int chunk = (N + blocks - 1) / blocks;
  if (chunk < THREADS) chunk = THREADS;
  blocks = (N + chunk - 1) / chunk;
  Seg* part = nullptr;
  cudaMalloc(&part, size_t(blocks) * sizeof(Seg));
  seg_partials<<<blocks, THREADS>>>(input, part, N, chunk);
  seg_final<<<1, 1>>>(part, blocks, output);
  cudaFree(part);
}
