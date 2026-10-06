# L1 — Launch, 1D/2D indexing, bounds, copies, error checks

**Weeks 1–6 (alongside P0–P1) · 3–4 problems/week ≈ 4 h/week · Tier T1 (LeetGPU), T2 optional via `harness/`**

**Objectives.**
1. Write `<<<grid, block>>>` launches with the ceil-div idiom.
2. Write grid-stride loops.
3. Bounds-check every access.
4. Map 2D `(row, col)` to row-major memory for any M×N.
5. Wrap every API call and launch in an error-check macro, and read `compute-sanitizer` output.
6. Transfer memory host↔device and know which calls synchronize.

**Problems.** See [leetgpu-map.md](leetgpu-map.md):
- core: 9 problems
- practice: 4 problems
- stretch: 2 problems

**Order:** Vector Addition → ReLU → Leaky ReLU → Color Inversion → Reverse Array → Matrix Copy → Matrix Addition → RGB to Grayscale → Matrix Multiplication (naive). Practice and stretch problems come after.

**Each solution includes:**
- the hint ladder:
  1. what one thread owns
  2. how to compute its index
  3. which bounds to check
- the kernel
- the harness test (fp32 exact or `rtol=1e-6`)
- one line on the memory traffic, in bytes moved per element

**Animations.** Lane A's P5.1 grid→blocks→warps animation is pulled forward as a reference.

**Exit check.** Unaided, write correct 2D indexing for a non-square, non-multiple-of-block shape (e.g. 1000×37). The harness runs it on 20 random shapes.

**Sources.**
- PMPP ch 2–3
- CUDA Programming Guide (programming model)
- srush GPU-Puzzles 1–8 as warm-up (linked)
