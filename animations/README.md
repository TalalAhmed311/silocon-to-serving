# Animations

These are first-class deliverables. Each one:

- is a single self-contained HTML file using vanilla JS + SVG/Canvas, with no network dependencies
- has play/pause/step controls and a speed slider
- works in light and dark mode and at phone width
- is embedded in the MkDocs site

All animations are written; none has been opened in a browser during the build (`TODO(run)`): check each once in light and dark mode and at phone width.

Static architecture and flow diagrams are written in Mermaid. Manim scenes are optional.

Where the Modular handbook already has a good interactive, we link it and build only what it lacks. See SOURCES.md §2.1.

| File | Module | Shows | Status |
|---|---|---|---|
| `p0-vm-page-walk.html` | P0.2 | VA → page table → TLB → physical page; page fault on first touch of an mmap'd file | built |
| `p0-false-sharing.html` | P0.3 | two cores on one cache line, MESI ping-pong, padded fix | built |
| `p0-simd-lanes.html` | P0.4 | scalar vs 8-wide AVX2 add; why misalignment costs | built |
| `roofline.html` | P0.4, P1.4 | drag arithmetic intensity; kernel moves from memory- to compute-bound | built |
| `p1-prefill-decode.html` | P1.2 | prefill vs decode token flow, growing KV cache, TTFT/ITL timeline | built |
| `p2-continuous-batching.html` | P2.3 | static vs continuous batching; chunked prefill letting decodes continue | built (complements handbook BatchingSimulator / ChunkedPrefillVisualizer) |
| `p2-paged-attention.html` | P2.3, P6.3 | logical → physical blocks, prefix sharing, copy-on-write | built |
| `p2-speculative-decoding.html` | P2.6, P6.5 | draft proposes k tokens, target verifies, accept/reject | built |
| `p3-request-path.html` | P3.4, P3.6 | gateway → queue → autoscaler → pod, cold vs warm | built |
| `p4-ring-allreduce.html` | P4.1 | ring all-reduce step by step | built |
| `p4-tp-mlp.html` | P4.2 | TP split of an MLP across 2/4 GPUs | built |
| `p5-grid-warps-sm.html` | P5.1 | grid → blocks → warps → SM scheduling; warp divergence | built (complements handbook GPUExecutionVisualizer / WarpDivergenceVisualizer) |
| `p5-coalescing.html` | P5.2 | contiguous vs strided access, sectors fetched | built (adds sector counts to handbook MemoryCoalescingVisualizer) |
| `p5-smem-tiling.html` | P5.2, P5.6 | shared-memory tiling of matmul; bank conflicts with/without padding | built |
| `p5-reduction.html` | P5.4 | tree reduction vs warp-shuffle reduction | built |
| `p5-online-softmax.html` | P5.5 | online softmax (running max and sum) | built |
| `p5-flashattention.html` | P5.8 | FlashAttention tiling over Q/K/V blocks | built |
| `p6-scheduler-loop.html` | P6.2 | waiting / running / preempted queues per step | built |
| `p6-radix-tree.html` | P6.4 | radix-tree prefix cache insert / match / evict | built |
| `p0-memory-layout.html` | P0.1 | struct padding; row- vs column-major strides (pilot animation) | built |
