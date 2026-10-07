// LeetGPU #3 Matrix Transpose — Lane B L2 solution (with the naive and unpadded variants for the bench).
// input: rows×cols row-major; output: cols×rows row-major. Device pointers.
#include <cuda_runtime.h>

constexpr int TILE = 32, ROWS_PER_PASS = 8;

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[size_t(c) * rows + r] = in[size_t(r) * cols + c];  // writes: stride `rows`
}

template <int PAD>
__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + PAD];  // PAD=1 shifts each row by one bank: column reads become conflict-free
  int c = blockIdx.x * TILE + threadIdx.x;
  int r0 = blockIdx.y * TILE + threadIdx.y;
  for (int k = 0; k < TILE; k += ROWS_PER_PASS)          // 32x8 threads cover a 32x32 tile in 4 passes
    if (r0 + k < rows && c < cols) tile[threadIdx.y + k][threadIdx.x] = in[size_t(r0 + k) * cols + c];
  __syncthreads();                                       // the whole tile must be loaded before anyone reads it
  int oc = blockIdx.y * TILE + threadIdx.x;              // output column = input row
  int or0 = blockIdx.x * TILE + threadIdx.y;             // output row = input column
  for (int k = 0; k < TILE; k += ROWS_PER_PASS)
    if (or0 + k < cols && oc < rows) out[size_t(or0 + k) * rows + oc] = tile[threadIdx.x][threadIdx.y + k];
}

void solve(const float* input, float* output, int rows, int cols) {
  dim3 block(TILE, ROWS_PER_PASS), grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transpose_tiled<1><<<grid, block>>>(input, output, rows, cols);
}
