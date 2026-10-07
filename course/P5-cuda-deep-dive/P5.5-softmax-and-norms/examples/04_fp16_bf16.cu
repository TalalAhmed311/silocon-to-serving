// 04_fp16_bf16.cu — why bf16 tolerances are ~1e-2 and fp16 ~1e-3: round-trip error of fp32 → T → fp32 over a range of
// magnitudes, plus the overflow difference (fp16 max ≈ 65504; bf16 keeps fp32's exponent range). Host-only demo of
// the cuda_fp16/cuda_bf16 conversions (runs anywhere CUDA headers are available; sm_75+ for the device part).
#include <cmath>
#include <cstdio>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

int main() {
  std::printf("| value | fp16 rel err | bf16 rel err |\n|---|---|---|\n");
  double worst16 = 0, worstbf = 0;
  for (double v : {1.0 + 1.0 / 3, 3.14159265, 1234.5678, 1e-3 * 7.77, 60000.0, 1e5, 3e38}) {
    const float f = float(v);
    const float h = __half2float(__float2half(f)), b = __bfloat162float(__float2bfloat16(f));
    const double eh = std::fabs((h - f) / f), eb = std::fabs((b - f) / f);
    std::printf("| %g | %s | %.2e |\n", v, std::isinf(h) ? "overflow (inf)" : (std::to_string(eh)).c_str(), eb);
    if (!std::isinf(h)) worst16 = std::fmax(worst16, eh);
    worstbf = std::fmax(worstbf, eb);
  }
  std::printf("worst observed: fp16 %.2e (unit roundoff 2^-11 = %.2e), bf16 %.2e (2^-8 = %.2e)\n",
              worst16, std::ldexp(1.0, -11), worstbf, std::ldexp(1.0, -8));
  std::printf("→ test tolerances: rtol ≈ a few × unit roundoff × (number of roundings on the path)\n");
}
