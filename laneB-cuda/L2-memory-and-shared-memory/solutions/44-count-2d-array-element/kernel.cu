// LeetGPU #44 Count 2D Array Element — Lane B L2 solution: a contiguous R×C matrix is a flat array.
#include "../43-count-array-element/kernel.cu"

void solve2d(const int* input, int* output, int R, int C, int K) {
  cudaMemset(output, 0, sizeof(int));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  count_eq<<<sms * 8, 256>>>(input, output, size_t(R) * C, K);
}
