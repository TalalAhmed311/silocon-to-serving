// s2s_ext.cu — CUDA implementation behind torch.ops.s2s.fused_add_rms_norm (#12), built with
// torch.utils.cpp_extension.load (JIT). Semantics match vLLM's fused_add_rms_norm (both tensors updated IN PLACE):
//   residual ← input + residual            (rounded to residual's dtype)
//   input    ← rmsnorm(residual) · weight   (fp32 math)
// Same algorithm as platform/kernels/include/d4/norms.cuh, without __restrict__ (input is read and then overwritten).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// c10::Half / c10::BFloat16 convert to and from float on the device; fp32 math throughout.
template <class T> __device__ __forceinline__ float tf(T v) { return static_cast<float>(v); }

__device__ __forceinline__ float block_sum(float v, float* scratch) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) scratch[w] = v;
  __syncthreads();
  v = lane < nw ? scratch[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <class T>
__global__ void fused_add_rmsnorm_inplace(T* x, T* resid, const T* w, int cols, float eps) {
  __shared__ float scratch[32];
  extern __shared__ float row[];
  const size_t base = (size_t)blockIdx.x * cols;
  float ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const T s = static_cast<T>(tf(x[base + c]) + tf(resid[base + c]));
    resid[base + c] = s;
    const float v = tf(s);
    row[c] = v;
    ss += v * v;
  }
  ss = block_sum(ss, scratch);
  const float r = rsqrtf(ss / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) x[base + c] = static_cast<T>(row[c] * r * tf(w[c]));
}

void fused_add_rms_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(), "contiguous tensors expected");
  TORCH_CHECK(x.sizes() == residual.sizes() && x.scalar_type() == residual.scalar_type(), "x/residual mismatch");
  const int cols = int(x.size(-1));
  const int rows = int(x.numel() / cols);
  TORCH_CHECK(cols <= 12288, "hidden size > 12288 needs the two-read variant (row buffer exceeds 48 KB)");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = cols >= 2048 ? 512 : 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "fused_add_rms_norm", [&] {
    fused_add_rmsnorm_inplace<scalar_t><<<rows, threads, cols * sizeof(float), stream>>>(
        x.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), cols, float(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_add_rms_norm", &fused_add_rms_norm, "in-place residual add + RMSNorm"); }
