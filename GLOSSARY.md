# Glossary

Each term is defined once, in one line. It is introduced in the module shown in brackets. Terms are listed in course order.

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
| KEDA | Kubernetes Event-driven Autoscaling: scales a Deployment on external metrics such as queue depth, including to zero. [P3.6] |
| Karpenter | A Kubernetes node autoscaler that launches right-sized (and spot) instances for pending pods. [P3.6] |
| Cold start | Time from scaling a replica up to serving its first request: node launch, image pull, weight load, warm-up. [P3.6] |
| Weight manifest | A list of a model's files with sizes and hashes, used to fetch, verify and cache weights. [P3.7] |
| NetworkPolicy | A Kubernetes rule set restricting which pods may talk to which; default-deny is the tenancy baseline. [P3.8] |
| Hash-chained audit log | An append-only log where each entry includes the hash of the previous one, so tampering is detectable. [P3.8] |
| α–β model | Communication time = per-message latency α + bytes ÷ bandwidth β; fitted from nccl-tests. [P4.1] |
| Ring all-reduce | All-reduce in 2(n−1) steps of S/n bytes around a ring of n ranks: reduce-scatter then all-gather. [P4.1] |
| Pipeline bubble | Idle time in pipeline parallelism while stages wait to fill and drain; (p−1)/(m+p−1) for 1F1B. [P4.2] |
| ZeRO / FSDP | Sharding optimizer state, gradients and parameters across data-parallel ranks, gathered on demand. [P4.2] |
| RTO / RPO | Recovery time objective (how long to recover) and recovery point objective (how much work or data may be lost). [P4.3] |
| Time-slicing / MPS | GPU sharing without hardware isolation: alternating contexts, or one shared context (Multi-Process Service). [P4.4] |
| Decoupled look-back | Single-pass parallel scan where each block publishes its aggregate and looks back at predecessors' flags. [P5.4] |
| Tensor core | A matrix-multiply-accumulate unit operating on small tiles (e.g. 16×16×16 fp16) per warp instruction. [P5.7] |
| cp.async | Ampere+ asynchronous global→shared copy that bypasses registers, enabling multi-stage pipelines. [P5.7] |
| FlashAttention | Tiled exact attention that keeps scores on chip with online softmax, never materialising the N×N matrix. [P5.8] |
| Custom op (torch.library) | A kernel registered with PyTorch's dispatcher so it works under torch.compile and in serving engines. [P5.10] |
| Token budget | The maximum number of tokens a scheduler places in one forward pass (max_num_batched_tokens). [P6.2] |
| Preemption by recompute | Freeing a running sequence's KV blocks under memory pressure and re-prefilling it later. [P6.2] |
| Aging | Raising a waiting request's effective priority with time so it cannot starve. [P6.2] |
| Block table | A sequence's list of physical KV-block ids; slot = table[pos // bs]·bs + pos % bs. [P6.3] |
| Slot mapping | The per-token list of cache slots a forward pass writes new K/V to. [P6.3] |
| Copy-on-write (KV) | Sharing KV blocks between forked sequences and copying a block only when one of them writes into it. [P6.3] |
| Top-p threshold search | Finding the nucleus cut-off by bisection on a probability threshold instead of sorting the vocabulary. [P6.5] |
| Rejection sampling (spec decode) | Accept draft token x with prob min(1, p(x)/q(x)); on rejection sample from max(p−q, 0); preserves p exactly. [P6.5] |
| Batch-size bucket | A padded batch size for which a CUDA graph was captured. [P6.6] |
| Hysteresis (health check) | Requiring k consecutive failures to mark down and more consecutive successes to mark up, to avoid flapping. [C1] |
| G-counter | A grow-only CRDT counter: one slot per replica, merged by element-wise max, summed for the value. [C1] |
| Failover routing (DNS) | Answering a name with the primary target while its health check passes, else the secondary. [C1] |
