// LeetGPU #7 Color Inversion — Lane B L1 solution. image: device pointer to width*height RGBA bytes, in place.
#include <cuda_runtime.h>

__global__ void invert(uchar4* __restrict__ px, int n_pixels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_pixels) {
    uchar4 p = px[i];               // one 4-byte load per pixel (cudaMalloc pointers are 256-byte aligned)
    p.x = 255 - p.x; p.y = 255 - p.y; p.z = 255 - p.z;  // alpha (p.w) untouched
    px[i] = p;
  }
}

void solve(unsigned char* image, int width, int height) {
  const int n = width * height, threads = 256;
  invert<<<(n + threads - 1) / threads, threads>>>(reinterpret_cast<uchar4*>(image), n);
}
