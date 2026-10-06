"""Exercise 2 solution."""


def ridge(peak_tflops, gbs):
    return peak_tflops * 1e12 / (gbs * 1e9)


def attainable_tflops(intensity, peak_tflops, gbs):
    return min(peak_tflops, intensity * gbs * 1e9 / 1e12)


def bound(intensity, peak_tflops, gbs):
    return "memory" if intensity < ridge(peak_tflops, gbs) else "compute"


def decode_intensity(batch, bytes_per_param=2):
    return 2 * batch / bytes_per_param   # each weight: 2 FLOPs per sequence, read once per step


def prefill_intensity(tokens, bytes_per_param=2):
    return 2 * tokens / bytes_per_param
