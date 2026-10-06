# Glossary

Each term is defined once, in one line. It is introduced in the module shown in brackets. This file grows as modules are built.

| Term | Definition |
|---|---|
| Arithmetic intensity | FLOPs performed per byte moved to or from memory; the x-axis of a roofline. [P0.4] |
| Roofline | A plot of attainable FLOP/s vs arithmetic intensity, capped by the memory-bandwidth slope and the compute peak. [P0.4] |
| Ridge point | The arithmetic intensity where the bandwidth slope meets the compute peak (peak FLOP/s ÷ bandwidth). [P0.4] |
| TLB | A small cache of virtual→physical page translations; a miss costs a page-table walk. [P0.2] |
| Page fault | A trap taken when a virtual page has no physical frame yet (minor: frame in page cache; major: needs disk I/O). [P0.2] |
| Pinned (page-locked) memory | Host memory the OS will not move or swap, so a DMA engine can read it directly. [P0.2] |
| False sharing | Two threads writing different variables on the same cache line, forcing the line to bounce between cores. [P0.3] |
| SIMD | Single instruction, multiple data: one instruction operates on a vector of lanes (AVX2: 8 × fp32). [P0.4] |
| Prefill | The first forward pass over the whole prompt; compute-bound, produces the KV cache and the first token. [P1.2] |
| Decode | Generating one token per step from the KV cache; memory-bound at small batch. [P1.2] |
| KV cache | Stored key and value tensors for past tokens, so attention does not recompute them. [P1.2] |
| TTFT | Time to first token: from request arrival to the first output token. [P1.3] |
| ITL | Inter-token latency: time between consecutive output tokens of one request. [P1.3] |
| TPOT | Time per output token: (end-to-end latency − TTFT) ÷ (output tokens − 1). [P1.3] |
| Goodput | Throughput counting only requests that met the SLO. [P1.3] |
| Continuous batching | Re-forming the batch at every decode step so finished requests leave and new ones join immediately. [P2.3] |
| Chunked prefill | Splitting a long prefill into chunks scheduled alongside decodes, so decodes are not stalled. [P2.3] |
| PagedAttention | Storing the KV cache in fixed-size blocks addressed through a per-sequence block table. [P2.3] |
| Prefix caching | Reusing KV blocks for a prompt prefix already computed for another request. [P2.3] |
| Speculative decoding | A cheap draft proposes k tokens; the target model verifies them in one pass, keeping the target's distribution. [P2.6] |
| Acceptance rate (α) | Fraction of draft tokens the target accepts. [P2.6] |
| CUDA graph | A captured sequence of GPU work replayed with one launch, removing per-kernel CPU launch overhead. [P2.6] |
| Bus bandwidth | nccl-tests' collective bandwidth normalized so it is comparable to the link's peak. [P4.1] |
| Tensor parallelism (TP) | Splitting each layer's weights across GPUs, with collectives inside every layer. [P4.2] |
| MIG | Multi-Instance GPU: hardware partitioning of A100/H100-class GPUs into isolated slices. [P4.4] |
| Warp | 32 threads that execute in lockstep on an SM. [P5.1] |
| Occupancy | Active warps per SM ÷ the maximum the SM supports. [P5.1] |
| Coalescing | Combining a warp's memory accesses into the fewest 32-byte sectors. [P5.2] |
| Bank conflict | Two threads of a warp hitting different addresses in the same shared-memory bank, which serializes the access. [P5.2] |
| Online softmax | Softmax computed in one pass by updating a running max and a rescaled running sum. [P5.5] |
| Radix tree (prefix cache) | A compressed trie over token IDs whose nodes own KV cache entries, used for longest-prefix reuse. [P6.4] |
