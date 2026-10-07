"""Exercise 1 solution."""
import numpy as np


def quantize(w, group=128, symmetric=True):
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group).astype(np.float64)
    if symmetric:
        scale = np.abs(g).max(axis=-1, keepdims=True) / 7.0
        scale[scale == 0] = 1.0                    # all-zero group: any scale works; avoid 0/0
        q = np.clip(np.round(g / scale), -8, 7)
        zero = np.zeros_like(scale)
    else:
        lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
        scale = (hi - lo) / 15.0
        scale[scale == 0] = 1.0
        zero = np.round(-lo / scale)
        q = np.clip(np.round(g / scale) + zero, 0, 15)
    return q.astype(np.int8), scale.astype(np.float32), zero.astype(np.float32), w.shape


def dequantize(q, scale, zero, shape):
    return ((q.astype(np.float32) - zero) * scale).reshape(shape)


def pack(q):
    u = (q.astype(np.int16) & 0xF).astype(np.uint8).reshape(-1)   # two's-complement nibble (works for 0..15 too)
    if u.size % 2:
        u = np.append(u, 0)
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8)


def unpack(packed, n, signed=True):
    lo, hi = packed & 0xF, packed >> 4
    u = np.stack([lo, hi], axis=1).reshape(-1)[:n].astype(np.int8)
    return np.where(u > 7, u - 16, u).astype(np.int8) if signed else u
