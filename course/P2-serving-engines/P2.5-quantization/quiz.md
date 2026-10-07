# P2.5 quiz

<details><summary><b>1.</b> Why does W4A16 speed up decode but not prefill?</summary>

Decode is memory-bound, so 4× fewer weight bytes means faster steps. Prefill is compute-bound, and the GEMM still runs in 16-bit, plus dequantization work.
</details>

<details><summary><b>2.</b> Why does FP8 dynamic quantization need no calibration data, but AWQ and GPTQ do?</summary>

FP8 has enough dynamic range that per-channel weight scales and per-token activation scales computed at runtime suffice. AWQ and GPTQ choose scales and rounding by measuring the error on real activations.
</details>

<details><summary><b>3.</b> What is the bits/weight of INT4 with group size 128 and one fp16 scale per group?</summary>

4 + 16/128 = **4.125** (plus a zero-point for asymmetric).
</details>

<details><summary><b>4.</b> E4M3 vs E5M2: which for inference weights and activations, and why?</summary>

**E4M3.** It has more mantissa (precision). Its range (±448) is enough with per-channel or per-token scales. E5M2's extra range is for gradients.
</details>

<details><summary><b>5.</b> One sentence each: how do AWQ and GPTQ differ?</summary>

GPTQ compensates the rounding error column by column using Hessian information. AWQ protects salient channels by scaling them before quantization.
</details>

<details><summary><b>6.</b> Why shouldn't you trust a 0.8-point accuracy difference measured on 250 samples?</summary>

The standard error of an accuracy p over n samples is √(p(1−p)/n), ≈ 2.5 points at p = 0.8 and n = 250. A 0.8-point difference is well inside the noise.
</details>
