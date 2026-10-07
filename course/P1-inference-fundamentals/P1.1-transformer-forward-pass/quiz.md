# P1.1 quiz

<details><summary><b>1.</b> Where does "≈ 2 FLOPs per parameter per token" come from?</summary>

Each weight is used in one multiply-add per token (a matvec), which is 2 FLOPs.
</details>

<details><summary><b>2.</b> A 7B fp16 model: bytes read per decode token at batch 1? Arithmetic intensity?</summary>

≈ 14 GB per token, and ≈ 2·7e9 FLOPs / 14e9 bytes = **1 FLOP/byte**.
</details>

<details><summary><b>3.</b> Why is prefill compute-bound when decode isn't, for the same model?</summary>

Prefill processes T tokens per weight read, so the intensity is ≈ T FLOP/byte (a GEMM). Decode at batch 1 uses each weight once per read.
</details>

<details><summary><b>4.</b> At what context length does attention's FLOP cost match the weight FLOPs for an 8B Llama (d = 4096, L = 32)?</summary>

t = 2P / (4·d·L) ≈ **31k tokens**.
</details>

<details><summary><b>5.</b> Does GQA reduce attention FLOPs?</summary>

No. Every query head still scores every key. GQA reduces KV-cache **size and read bytes** by H/H_kv.
</details>

<details><summary><b>6.</b> Why test the NumPy reference against HF rather than trust it?</summary>

Every downstream implementation (C++, CUDA, custom kernels) is tested against it. A bug in the reference propagates silently into all of them.
</details>

<details><summary><b>7.</b> Is the embedding lookup counted in FLOPs/token?</summary>

No. It copies one row (d values). The LM head *is* a full V×d matvec.
</details>
