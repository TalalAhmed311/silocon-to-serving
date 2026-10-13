// 02_transfer_tax.cu: time the three parts of a vector-scale program with CUDA events, and the one-time
// start-up cost with a host clock. Compare the printed GB/s with the lesson's "transfer tax" simulator and with
// the spec bandwidth in P1.4's gpu_specs.yaml. Needs an NVIDIA GPU. TODO(run-on: T2)
#include <chrono>
#include <cstdio>
#include <vector>

#include "s2s_cuda.cuh"

__global__ void scale_kernel(float* x, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= factor;
}

int main() {
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  CUDA_CHECK(cudaFree(nullptr));  // the first CUDA call creates the context; time it on the host clock
  double init_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  std::printf("first CUDA call (context creation): %.1f ms on the host clock\n", init_ms);

  const int n = 1 << 26;  // 67,108,864 floats = 256 MiB
  const size_t bytes = size_t(n) * sizeof(float);
  std::vector<float> h(static_cast<size_t>(n), 1.0f);  // pageable host memory (P0.2): see P5.1 for pinned memory
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));

  cudaEvent_t e0, e1, e2, e3;
  for (cudaEvent_t* e : {&e0, &e1, &e2, &e3}) CUDA_CHECK(cudaEventCreate(e));
  const int threads = 256, blocks = (n + threads - 1) / threads;

  // Run the sequence twice; the first pass pays one-time costs (module load, page faults on the host buffer).
  for (int pass = 0; pass < 2; ++pass) {
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    scale_kernel<<<blocks, threads>>>(d, n, 2.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));
    float h2d = 0, kern = 0, d2h = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&kern, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, e2, e3));
    // GB/s = bytes / (ms * 1e6). The kernel reads and writes every element: 2 * bytes.
    std::printf("pass %d: H2D %.2f ms (%.1f GB/s) | kernel %.3f ms (%.1f GB/s) | D2H %.2f ms (%.1f GB/s)\n", pass, h2d,
                double(bytes) / (h2d * 1e6), kern, 2.0 * double(bytes) / (kern * 1e6), d2h, double(bytes) / (d2h * 1e6));
  }
  std::printf("h[0] = %g (2 passes of x2 on 1.0 = 4)\n", h[0]);
  for (cudaEvent_t e : {e0, e1, e2, e3}) cudaEventDestroy(e);
  CUDA_CHECK(cudaFree(d));
  return 0;
}
