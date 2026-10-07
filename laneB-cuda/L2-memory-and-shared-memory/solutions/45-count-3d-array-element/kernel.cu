// LeetGPU #45 Count 3D Array Element — Lane B L2 solution.
#include "../43-count-array-element/kernel.cu"

void solve3d(const int* input, int* output, int D, int R, int C, int K) {
  cudaMemset(output, 0, sizeof(int));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  count_eq<<<sms * 8, 256>>>(input, output, size_t(D) * R * C, K);
}
