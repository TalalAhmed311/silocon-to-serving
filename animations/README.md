# Animations

These are first-class deliverables. Each one:

- is a single self-contained HTML file using vanilla JS + SVG/Canvas, with no network dependencies
- has play/pause/step controls and a speed slider
- works in light and dark mode and at phone width
- is embedded in the MkDocs site

Static architecture and flow diagrams are written in Mermaid. Manim scenes are optional.

Where the Modular handbook already has a good interactive, we link it and build only what it lacks. See SOURCES.md §2.1.

| File | Module | Shows | Status |
|---|---|---|---|
| `p0-vm-page-walk.html` | P0.2 | VA → page table → TLB → physical page; page fault on first touch of an mmap'd file | planned |
| `p0-false-sharing.html` | P0.3 | two cores on one cache line, MESI ping-pong, padded fix | planned |
| `p0-simd-lanes.html` | P0.4 | scalar vs 8-wide AVX2 add; why misalignment costs | planned |
| `roofline.html` | P0.4, P1.4 | drag arithmetic intensity; kernel moves from memory- to compute-bound | planned |
| `p1-prefill-decode.html` | P1.2 | prefill vs decode token flow, growing KV cache, TTFT/ITL timeline | planned |
| `p2-continuous-batching.html` | P2.3 | static vs continuous batching; chunked prefill letting decodes continue | planned (complements handbook BatchingSimulator / ChunkedPrefillVisualizer) |
| `p2-paged-attention.html` | P2.3, P6.3 | logical → physical blocks, prefix sharing, copy-on-write | planned |
| `p2-speculative-decoding.html` | P2.6, P6.5 | draft proposes k tokens, target verifies, accept/reject | planned |
| `p3-request-path.html` | P3.4, P3.6 | gateway → queue → autoscaler → pod, cold vs warm | planned |
| `p4-ring-allreduce.html` | P4.1 | ring all-reduce step by step | planned |
| `p4-tp-mlp.html` | P4.2 | TP split of an MLP across 2/4 GPUs | planned |
| `p5-grid-warps-sm.html` | P5.1 | grid → blocks → warps → SM scheduling; warp divergence | planned (complements handbook GPUExecutionVisualizer / WarpDivergenceVisualizer) |
| `p5-coalescing.html` | P5.2 | contiguous vs strided access, sectors fetched | planned (adds sector counts to handbook MemoryCoalescingVisualizer) |
| `p5-smem-tiling.html` | P5.2, P5.6 | shared-memory tiling of matmul; bank conflicts with/without padding | planned |
| `p5-reduction.html` | P5.4 | tree reduction vs warp-shuffle reduction | planned |
| `p5-online-softmax.html` | P5.5 | online softmax (running max and sum) | planned |
| `p5-flashattention.html` | P5.8 | FlashAttention tiling over Q/K/V blocks | planned |
| `p6-scheduler-loop.html` | P6.2 | waiting / running / preempted queues per step | planned |
| `p6-radix-tree.html` | P6.4 | radix-tree prefix cache insert / match / evict | planned |
| `p0-memory-layout.html` | P0.1 | struct padding; row- vs column-major strides (pilot animation) | planned for Stage 3 |
