"""Exercise 2 solution. Intensity is monotone increasing in B, so a linear scan (or a closed form) works."""


def crossover_batch(P, d, L, kv_bytes_per_token, t, peak_flops, bw, bytes_per_param=2, max_batch=4096):
    ridge = peak_flops / bw
    for B in range(1, max_batch + 1):
        flops = 2 * P * B + B * 4 * d * L * t
        bytes_ = P * bytes_per_param + B * t * kv_bytes_per_token
        if flops / bytes_ >= ridge:
            return B
    return None
    # Limit B -> inf: intensity -> (2P + 4dLt) / (t·kv) — a constant set by the KV cache, independent of B.
