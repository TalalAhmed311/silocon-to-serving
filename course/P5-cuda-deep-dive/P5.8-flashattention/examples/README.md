# P5.8 examples (T2)

| File | What |
|---|---|
| [`attention_bench.cu`](attention_bench.cu) | naive vs FA-1 vs FA-2 (`d4/attention.cuh`), full and causal, N = 1k–4k: the L6 exit-check table |
| [`vs_flash_attn.py`](vs_flash_attn.py) | PyTorch SDPA and `flash-attn v2.8.3.post1` on the same shapes |
| [`attn_ref.hpp`](attn_ref.hpp) | the double-precision CPU reference the exercise tests share |

Reading: tspeterkim/flash-attention-minimal `flash.cu` (~100 lines; FA-1 structure; read and annotate, don't copy). Then the FA-1 and FA-2 papers' Algorithm 1.
