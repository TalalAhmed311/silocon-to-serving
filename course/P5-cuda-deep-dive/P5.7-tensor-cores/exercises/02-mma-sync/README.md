# Exercise 2: `ldmatrix` + `mma.sync` (sm_80+, T2)

WMMA hides the fragment layout. `mma.sync` exposes it. `kernel.cuh` gives you the tile loader and the PTX wrappers. Write:

1. **The fragment loop.** For `mma.m16n8k16.row.col`, A is a 16×16 row-major fragment in 4 registers (8 halves). `ldmatrix.x4` loads it in one instruction: lane `l` supplies the shared-memory address of row `l % 16`, column `8·(l / 16)`, and the four 8×8 pieces land in the order the PTX ISA's A layout needs. B is 16×8 "col". Since our B tile is row-major K×N, use `ldmatrix.x2.trans`, with lane `l < 16` supplying the address of row `k = l`.
2. **The store.** Accumulator registers `c0, c1` hold row `g = lane/4`, columns `2t, 2t+1` (`t = lane%4`). `c2, c3` hold row `g + 8`.

Keep the PTX ISA *Matrix fragments for mma.m16n8k16* figures open. Every bug here is a layout bug, and the test (same tolerance as exercise 1) will catch it. Then compare your kernel with WMMA on an L4: same block and warp tiles, so how close are they, and why? (The compiler generates similar SASS. Inspect it with `cuobjdump -sass`.)
