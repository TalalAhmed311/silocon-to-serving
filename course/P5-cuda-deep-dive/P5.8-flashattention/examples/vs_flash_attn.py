"""vs_flash_attn.py — PyTorch SDPA and the pinned flash-attn package on the same shapes as attention_bench (T2).

Run (GPU box): uv pip install torch flash-attn==2.8.3.post1 --no-build-isolation   # pinned per SOURCES.md
               uv run python course/P5-cuda-deep-dive/P5.8-flashattention/examples/vs_flash_attn.py
Prints | N | causal | SDPA ms | flash_attn ms | — compare with attention_bench's FA-2 column. flash-attn needs sm_80+
(an L4 works; a T4 doesn't: the script then reports SDPA only). Layout notes: flash_attn_func takes [B, N, H, D].
"""
import torch


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def main():
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        flash_attn_func = None
    B, H, D = 1, 16, 64
    print("| N | causal | SDPA ms | flash_attn ms |\n|---|---|---|---|")
    for N in (1024, 2048, 4096):
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
        for causal in (False, True):
            sd = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal))
            fa = "—"
            if flash_attn_func is not None and torch.cuda.get_device_capability() >= (8, 0):
                qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
                fa = f"{bench(lambda: flash_attn_func(qt, kt, vt, causal=causal)):.3f}"
            print(f"| {N} | {causal} | {sd:.3f} | {fa} |")


if __name__ == "__main__":
    main()
