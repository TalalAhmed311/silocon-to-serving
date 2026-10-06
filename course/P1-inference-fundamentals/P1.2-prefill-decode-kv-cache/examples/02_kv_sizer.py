"""02_kv_sizer.py — KV-cache bytes per token / per sequence and how many sequences fit next to the weights.

Run:      uv run python course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/examples/02_kv_sizer.py \
            --layers 32 --kv-heads 8 --head-dim 128 --gpu-gb 24 --weights-gb 16 [--overhead-gb 1.5]
Expected: exact arithmetic from your inputs (nothing measured).
Hardware: T0.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, required=True)
ap.add_argument("--kv-heads", type=int, required=True)
ap.add_argument("--head-dim", type=int, required=True)
ap.add_argument("--gpu-gb", type=float, required=True)
ap.add_argument("--weights-gb", type=float, required=True)
ap.add_argument("--overhead-gb", type=float, default=1.5, help="activations + CUDA context + allocator slack (estimate)")
a = ap.parse_args()

free = (a.gpu_gb - a.weights_gb - a.overhead_gb) * 2**30   # GiB throughout (GPU memory is sized in GiB)
print(f"KV budget: {free / 2**30:.2f} GiB\n")
print("| KV dtype | bytes/token | 1k-token seq | 8k-token seq | tokens that fit | 2k-token seqs that fit |")
print("|---|---|---|---|---|---|")
for name, b in (("fp16/bf16", 2), ("fp8", 1)):
    per_tok = 2 * a.layers * a.kv_heads * a.head_dim * b
    fit = int(free // per_tok)
    print(f"| {name} | {per_tok:,} | {per_tok * 1024 / 2**20:.0f} MiB | {per_tok * 8192 / 2**30:.2f} GiB | {fit:,} | {fit // 2048} |")
