"""Exercise 4 solution."""
import torch


def measure_copy_gbs(n_bytes: int) -> float:
    n = n_bytes // 4
    a, b = torch.empty(n, device="cuda"), torch.empty(n, device="cuda")
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    ms = []
    for _ in range(20):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); b.copy_(a); e.record(); e.synchronize()
        ms.append(s.elapsed_time(e))
    return 2 * n * 4 / (sorted(ms)[10] * 1e6)   # read + write bytes / seconds -> GB/s
