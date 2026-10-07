# 7 — Color Inversion (L1)

Problem: [LeetGPU #7](../../leetgpu-map.md). The image is a flat array of RGBA bytes: `width × height` pixels, 4 bytes each. Invert R, G and B (`255 − v`) and leave alpha alone.

**Hint ladder**

1. Should one thread own one byte or one pixel? Pick a pixel: then the "skip alpha" logic disappears.
2. Pixel `p` starts at byte `4p`. Index `p` over `width × height`, not over bytes.
3. Faster: reinterpret the buffer as `uchar4*`, or as a `uint32_t*` and XOR with `0x00FFFFFF`, which flips the low 3 bytes on little-endian. That is one 4-byte load and one store per pixel.

**Solution outline:** one thread per pixel, a `uchar4` load, invert `.x .y .z`, store.

**Why this is fast:** a 4-byte access per thread means each warp moves 128 contiguous bytes, fully coalesced. Byte-per-thread versions issue 4× more instructions for the same bytes. Memory-bound, so compare with the copy row in `--bench`.
