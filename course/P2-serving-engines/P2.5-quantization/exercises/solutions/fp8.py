"""Exercise 2 solution: table-driven round-to-nearest-even into FP8 E4M3 (fn) / E5M2, with saturation."""
import numpy as np

_FMT = {"e4m3": (4, 3, 7), "e5m2": (5, 2, 15)}


def _decode(code: int, fmt: str) -> float:
    e_bits, m_bits, bias = _FMT[fmt]
    exp = (code >> m_bits) & ((1 << e_bits) - 1)
    man = code & ((1 << m_bits) - 1)
    if fmt == "e4m3" and exp == 0xF and man == 0x7:
        return np.nan                                   # E4M3fn: only S.1111.111 is NaN; no infinities
    if fmt == "e5m2" and exp == 0x1F:
        return np.inf if man == 0 else np.nan
    if exp == 0:
        return man * 2.0 ** (1 - bias - m_bits)        # subnormal
    return (1 + man / 2**m_bits) * 2.0 ** (exp - bias)


def _table(fmt):
    codes = np.arange(0, 128)                           # sign bit 0: non-negative values, increasing with code
    vals = np.array([_decode(int(c), fmt) for c in codes])
    keep = np.isfinite(vals)
    return vals[keep], codes[keep]


def representable(fmt="e4m3"):
    return _table(fmt)[0]


def quantize(x, fmt="e4m3"):
    vals, codes = _table(fmt)
    a = np.minimum(np.abs(np.asarray(x, dtype=np.float64)), vals[-1])   # saturate to max finite
    hi = np.clip(np.searchsorted(vals, a), 1, len(vals) - 1)
    lo = hi - 1
    dlo, dhi = a - vals[lo], vals[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (codes[hi] % 2 == 0))      # ties -> even code
    out = np.where(pick_hi, vals[hi], vals[lo])
    return (np.sign(x) * out).astype(np.float32)
