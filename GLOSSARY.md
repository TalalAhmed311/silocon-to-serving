# Glossary

Each term is defined once, in one line. It is introduced in the module shown in brackets. Terms are listed in course order.

| Term | Definition |
|---|---|
| Object file | The compiler's output for one source file: machine code with unresolved references to other files. [R.1] |
| Linker | The tool that joins object files and libraries into one executable by resolving those references. [R.1] |
| Pointer | A variable that holds a memory address; its type says how many bytes `p + 1` steps. [R.1] |
| Reference (C++) | An alias for an existing object. It cannot be null or re-seated. [R.1] |
| Undefined behaviour | Code the language gives no meaning to (out-of-bounds read, uninitialised use); the compiler may do anything. [R.1] |
| Stack frame | The block of stack memory holding one function call's locals; freed automatically on return. [R.2] |
| Heap | Memory you allocate and free yourself (`new`/`delete`, `malloc`/`free`); its lifetime is up to you. [R.2] |
| Dangling pointer | A pointer to memory that has been freed or has gone out of scope. [R.2] |
| Memory leak | Heap memory that is never freed after its last use. [R.2] |
| Use-after-free | Reading or writing a heap block after it has been freed. [R.2] |
| RAII | Resource Acquisition Is Initialization: acquire in the constructor, release in the destructor, so cleanup follows scope. [R.2] |
| AddressSanitizer (ASan) | A compiler option (`-fsanitize=address`) that checks memory accesses at run time and reports overflows and use-after-free. [R.2] |
| `std::unique_ptr` | A smart pointer that is the sole owner of a heap object, frees it on destruction and can only be moved. [R.2] |
| Exit code | The small integer a program returns to the shell (`$?`); zero means success. [R.3] |
| Pipe | `a \| b`: connects the output of one process to the input of the next. [R.3] |
| Environment variable | A named string a process inherits from its parent, such as `S2S_SOLUTIONS=1`. [R.3] |
| Staging area (index) | Where `git add` collects changes that the next commit will record. [R.3] |
| ndarray | NumPy's array: a typed flat buffer plus a shape and strides. [R.4] |
| dtype | The element type of an array (float32, float16, int8); it sets the bytes per element. [R.4] |
| Stride | The number of bytes to step to reach the next element along an axis. [R.4] |
| View (NumPy) | An array that shares another array's memory, such as a slice or a transpose. [R.4] |
| Broadcasting | NumPy's rule for combining shapes: align them from the right and stretch size-1 dimensions. [R.4] |
| Logit | A raw, unnormalised score the model gives each vocabulary entry. [R.5] |
| Softmax | Turns scores into probabilities: exp of each, divided by the sum. The stable form subtracts the max first. [R.5] |
| Temperature | A divisor applied to logits before softmax; below 1 sharpens the distribution, above 1 flattens it. [R.5] |
| Token id | The integer index of a token in the vocabulary. [R.5] |
| HTTP status code | The three-digit result of a request: 2xx success, 4xx client error, 5xx server error. [R.6] |
| SSE (server-sent events) | A one-way stream over HTTP where the server sends `data: …` events, each ended by a blank line. [R.6] |
| Idempotent | An operation that has the same effect if repeated, so retrying it is safe. [R.6] |
| IAM Identity Center | AWS's single sign-on service: short-lived logins for people, instead of long-lived access keys. [R.7] |
| Budget alarm | A cost threshold on the account that sends an alert, such as an AWS Budgets email. [R.7] |
| Session Manager (SSM) | AWS access to an instance's shell through the AWS API, with no inbound SSH port. [R.7] |
| Terraform state | Terraform's record of the resources it created; `plan` compares it with your files. [R.7] |
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
| Host and device | In CUDA, the CPU with its RAM (host) and the GPU with its memory (device), joined by PCIe. [P1.6] |
| `nvcc` | NVIDIA's CUDA compiler driver: sends host code to the C++ compiler and device code to PTX/SASS. [P1.6] |
| Compute capability | A version number (such as 7.5, passed as `-arch=sm_75`) naming the features a GPU supports. [P1.6] |
| Kernel launch | `kernel<<<grid, block>>>(args)`: starts a grid of blocks of threads; it returns before the kernel finishes. [P1.6] |
| `cudaMemcpy` | Copies bytes between host and device memory; its direction flag says which way. [P1.6] |
| compute-sanitizer | NVIDIA's tool that runs a CUDA program and reports out-of-bounds and misaligned memory accesses. [P1.6] |
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
| Image layer | One read-only step of a container image; layers are cached and shared, and a changed layer invalidates those after it. [P3.0] |
| Container | A process started from an image, isolated by namespaces and limited by cgroups, with a writable layer on top. [P3.0] |
| cgroup | A Linux kernel feature that limits and accounts the CPU, memory and other resources of a group of processes. [P3.0] |
| Image digest | A content hash (`sha256:…`) that names one exact image; unlike a tag, it cannot move. [P3.0] |
| Reconciliation loop | A controller repeatedly comparing desired state with actual state and acting to close the gap. [P3.0] |
| Pod | Kubernetes' smallest unit: one or more containers that share a network and are scheduled together. [P3.0] |
| Deployment | A Kubernetes object that keeps a set number of identical Pod replicas running and rolls out updates. [P3.0] |
| Service (Kubernetes) | A stable name and address that routes traffic to the Pods matching a label selector. [P3.0] |
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
