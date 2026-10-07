# PROGRESS

Tick each box as you finish it. Lane A ≈ 11 h/week. Lane B does 3–4 problems/week in weeks 1–24, then daily from week 25. Phase boundaries follow the curriculum, and the split into weeks inside a phase is a suggestion.

| Week | Lane A | Lane B | Done |
|---|---|---|---|
| 1 | P0.1 C/C++ for systems | L1 · 3–4 problems | ☐ |
| 2 | P0.2 Virtual memory · P0.3 Threads/atomics (D2) | L1 · 3–4 problems | ☐ |
| 3 | P0.4 SIMD + roofline (D1) · P0.5 #0 v0 | L1 · 3–4 problems | ☐ |
| 4 | P1.1 Transformer forward pass · P1.2 Prefill/decode | L1 · 3–4 problems | ☐ |
| 5 | P1.3 Metrics · P1.4 GPU roofline | L1 · 3–4 problems | ☐ |
| 6 | P1.5 D3 capacity calculator | L1 · 3–4 problems | ☐ |
| 7 | P2.1 First serve with vLLM | L2 · 3–4 problems | ☐ |
| 8 | P2.2 SGLang · P2.4 Benchmarking | L2 · 3–4 problems | ☐ |
| 9 | P2.3 Batching + #4 load test | L2 · 3–4 problems | ☐ |
| 10 | P2.5 Quantization + #7 | L2 · 3–4 problems | ☐ |
| 11 | P2.6 Spec decoding + CUDA graphs + #10 | L2 · 3–4 problems | ☐ |
| 12 | P2.7 Gateway #6 | L2 · 3–4 problems | ☐ |
| 13 | P3.1 GPU containers | L2 · 3–4 problems | ☐ |
| 14 | P3.2 Terraform on AWS | L2 · 3–4 problems | ☐ |
| 15 | P3.3 K8s + GPU Operator | L3 · 3–4 problems | ☐ |
| 16 | P3.4 Serving stacks + #1 | L3 · 3–4 problems | ☐ |
| 17 | P3.5 Observability #13 + #2 | L3 · 3–4 problems | ☐ |
| 18 | P3.6 Autoscaling #3 | L3 · 3–4 problems | ☐ |
| 19 | P3.7 Weight delivery #5 · P3.8 Tenancy #8 | L3 · 3–4 problems | ☐ |
| 20 | P3.9 Second cloud | L3 · 3–4 problems | ☐ |
| 21 | P4.1 NCCL | L3 · 3–4 problems | ☐ |
| 22 | P4.2 Parallelism | L3 · 3–4 problems | ☐ |
| 23 | P4.3 Distributed jobs #9 | L3 · 3–4 problems | ☐ |
| 24 | P4.4 GPU sharing #11 | L3 · 3–4 problems | ☐ |
| 25 | P5.1 Execution model | L4 · daily | ☐ |
| 26 | P5.2 Memory | L4 · daily | ☐ |
| 27 | P5.3 Nsight | L4 · daily | ☐ |
| 28 | P5.4 Reductions/scans | L5 · daily | ☐ |
| 29 | P5.5 Softmax/norms | L5 · daily | ☐ |
| 30 | P5.6 GEMM ladder (1/2) | L5 · daily | ☐ |
| 31 | P5.6 GEMM ladder (2/2) | L5 · daily | ☐ |
| 32 | P5.7 Tensor cores | L6 · daily | ☐ |
| 33 | P5.8 FlashAttention | L6 · daily | ☐ |
| 34 | P5.9 Triton · P5.10 #12 | L6 · daily | ☐ |
| 35 | P6.1 Reading nano-vllm | L6 · daily | ☐ |
| 36 | P6.2 Scheduler | L6 · daily | ☐ |
| 37 | P6.3 Paged KV | L6 · daily | ☐ |
| 38 | P6.4 Radix prefix cache · P6.5 Spec verify + sampling | L6 · daily | ☐ |
| 39 | P6.6 CUDA graphs · P6.7 #0 v1 (1/2) | L6 · daily | ☐ |
| 40 | P6.7 #0 v1 (2/2) | L6 · daily | ☐ |
| 41 | C1 Failover #14 (1/2) | L6 · daily | ☐ |
| 42 | C1 Failover #14 (2/2) | L6 · daily | ☐ |
| 43 | C2 Teardown #15 (1/2) | L6 · daily | ☐ |
| 44 | C2 Teardown #15 (2/2) | L6 · daily | ☐ |

## Exit checks

- ☐ L1: 2D indexing for any shape, unaided
- ☐ L2: transpose ≥80% of copy bandwidth
- ☐ L3: reduction within 10% of `cub::DeviceReduce`
- ☐ L4: single-pass online softmax, reported in GB/s
- ☐ L5: SGEMM ≥70% cuBLAS; tensor-core HGEMM ≥50%
- ☐ L6: FA-2 forward ≥5× naive at seq 4k
