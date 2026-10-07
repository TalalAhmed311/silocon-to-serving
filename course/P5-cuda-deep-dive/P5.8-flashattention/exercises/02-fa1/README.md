# Exercise 2: FlashAttention-1 structure (T2)

Read tspeterkim's `flash.cu` (FA-1 structure, ~100 lines) and the FA-1 paper's Algorithm 1. Annotate them, but **don't copy** them. Then implement `attention` in `kernel.cuh`:

- **outer loop over K/V tiles** (Bc keys): load the tile into shared memory once
- for each query row: scores against the tile → tile max → update running `m` and `l` → rescale and accumulate `O`
- `m`, `l` and the unnormalized `O` live in **global** workspaces between K/V tiles (FA-1 keeps them in HBM). Normalize by `l` at the end

The test is the same as exercise 1's. Then compare its time with exercise 1 and with FA-2 (exercise 3). The FA-2 paper's first change was exactly this loop order. How much HBM traffic does FA-1 spend on re-reading and re-writing `O`?
