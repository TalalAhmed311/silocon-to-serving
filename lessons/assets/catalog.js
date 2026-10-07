/* catalog.js — every lesson in the course, in order. `id` names lessons/<id>.html and lessons/content/<id>.js.
   `src` is the module folder whose exercises and labs form the lesson's Practice tab. */
window.S2S_CATALOG = [
  { phase: "P0", title: "Systems primer", weeks: "1–3", blurb: "How a computer really runs your code: memory, the OS, threads, caches and SIMD.", lessons: [
    { id: "p0-1", n: "P0.1", title: "C and C++ for systems work", src: "course/P0-systems-primer/P0.1-c-cpp-for-systems" },
    { id: "p0-2", n: "P0.2", title: "Virtual memory and mmap", src: "course/P0-systems-primer/P0.2-virtual-memory-and-mmap" },
    { id: "p0-3", n: "P0.3", title: "Threads, atomics and caches", src: "course/P0-systems-primer/P0.3-threads-atomics-caches" },
    { id: "p0-4", n: "P0.4", title: "SIMD and the roofline", src: "course/P0-systems-primer/P0.4-simd-and-roofline" },
    { id: "p0-5", n: "P0.5", title: "Project: a tiny CPU inference engine", src: "course/P0-systems-primer/P0.5-project-tiny-engine-cpu" } ] },
  { phase: "P1", title: "Inference fundamentals", weeks: "4–6", blurb: "What happens inside one forward pass, why decoding is slow, and how to size hardware.", lessons: [
    { id: "p1-1", n: "P1.1", title: "The transformer forward pass", src: "course/P1-inference-fundamentals/P1.1-transformer-forward-pass" },
    { id: "p1-2", n: "P1.2", title: "Prefill, decode and the KV cache", src: "course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache" },
    { id: "p1-3", n: "P1.3", title: "Latency and throughput metrics", src: "course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics" },
    { id: "p1-4", n: "P1.4", title: "GPU architecture and the roofline", src: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline" },
    { id: "p1-5", n: "P1.5", title: "Project: a capacity calculator", src: "course/P1-inference-fundamentals/P1.5-project-capacity-calculator" } ] },
  { phase: "P2", title: "Serving engines", weeks: "7–12", blurb: "Run real engines, batch requests, quantize, speculate, and put a gateway in front.", lessons: [
    { id: "p2-1", n: "P2.1", title: "First serve with vLLM", src: "course/P2-serving-engines/P2.1-first-serve-with-vllm" },
    { id: "p2-2", n: "P2.2", title: "SGLang and its runtime", src: "course/P2-serving-engines/P2.2-sglang" },
    { id: "p2-3", n: "P2.3", title: "Batching and caching", src: "course/P2-serving-engines/P2.3-batching-and-caching" },
    { id: "p2-4", n: "P2.4", title: "Benchmarking methodology", src: "course/P2-serving-engines/P2.4-benchmarking-methodology" },
    { id: "p2-5", n: "P2.5", title: "Quantization", src: "course/P2-serving-engines/P2.5-quantization" },
    { id: "p2-6", n: "P2.6", title: "Speculative decoding and CUDA graphs", src: "course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs" },
    { id: "p2-7", n: "P2.7", title: "Gateway basics", src: "course/P2-serving-engines/P2.7-gateway-basics" } ] },
  { phase: "P3", title: "Deployment and infrastructure", weeks: "13–20", blurb: "Containers, Terraform, Kubernetes with GPUs, observability, autoscaling, tenancy and a second cloud.", lessons: [
    { id: "p3-1", n: "P3.1", title: "GPU containers", src: "course/P3-deployment-and-infra/P3.1-gpu-containers" },
    { id: "p3-2", n: "P3.2", title: "Terraform on AWS", src: "course/P3-deployment-and-infra/P3.2-terraform-on-aws" },
    { id: "p3-3", n: "P3.3", title: "Kubernetes and the GPU Operator", src: "course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator" },
    { id: "p3-4", n: "P3.4", title: "Serving stacks on Kubernetes", src: "course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s" },
    { id: "p3-5", n: "P3.5", title: "Observability", src: "course/P3-deployment-and-infra/P3.5-observability" },
    { id: "p3-6", n: "P3.6", title: "Autoscaling GPUs", src: "course/P3-deployment-and-infra/P3.6-autoscaling" },
    { id: "p3-7", n: "P3.7", title: "Weight delivery", src: "course/P3-deployment-and-infra/P3.7-weight-delivery" },
    { id: "p3-8", n: "P3.8", title: "Multi-tenancy and security", src: "course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security" },
    { id: "p3-9", n: "P3.9", title: "A second cloud", src: "course/P3-deployment-and-infra/P3.9-second-cloud" } ] },
  { phase: "P4", title: "Multi-GPU and distributed", weeks: "21–24", blurb: "How GPUs talk, how models are split across them, and how to share one GPU fairly.", lessons: [
    { id: "p4-1", n: "P4.1", title: "NCCL collectives", src: "course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives" },
    { id: "p4-2", n: "P4.2", title: "Parallelism strategies", src: "course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies" },
    { id: "p4-3", n: "P4.3", title: "Distributed jobs and checkpointing", src: "course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing" },
    { id: "p4-4", n: "P4.4", title: "GPU sharing", src: "course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing" } ] },
  { phase: "P5", title: "CUDA deep dive", weeks: "25–34", blurb: "Write fast GPU kernels: the execution model, memory, reductions, GEMM, tensor cores, FlashAttention, Triton.", lessons: [
    { id: "p5-1", n: "P5.1", title: "The CUDA execution model", src: "course/P5-cuda-deep-dive/P5.1-execution-model" },
    { id: "p5-2", n: "P5.2", title: "Coalescing and shared memory", src: "course/P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory" },
    { id: "p5-3", n: "P5.3", title: "Profiling with Nsight", src: "course/P5-cuda-deep-dive/P5.3-profiling-with-nsight" },
    { id: "p5-4", n: "P5.4", title: "Reductions and scans", src: "course/P5-cuda-deep-dive/P5.4-reductions-and-scans" },
    { id: "p5-5", n: "P5.5", title: "Softmax and norms", src: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms" },
    { id: "p5-6", n: "P5.6", title: "The GEMM ladder", src: "course/P5-cuda-deep-dive/P5.6-gemm-ladder" },
    { id: "p5-7", n: "P5.7", title: "Tensor cores", src: "course/P5-cuda-deep-dive/P5.7-tensor-cores" },
    { id: "p5-8", n: "P5.8", title: "FlashAttention", src: "course/P5-cuda-deep-dive/P5.8-flashattention" },
    { id: "p5-9", n: "P5.9", title: "Triton", src: "course/P5-cuda-deep-dive/P5.9-triton" },
    { id: "p5-10", n: "P5.10", title: "A custom op inside a serving engine", src: "course/P5-cuda-deep-dive/P5.10-custom-op-into-serving" } ] },
  { phase: "P6", title: "Engine internals", weeks: "35–40", blurb: "Rebuild a serving engine: scheduler, paged KV, prefix cache, sampling, CUDA graphs.", lessons: [
    { id: "p6-1", n: "P6.1", title: "Reading nano-vllm", src: "course/P6-engine-internals/P6.1-reading-nano-vllm" },
    { id: "p6-2", n: "P6.2", title: "The scheduler", src: "course/P6-engine-internals/P6.2-scheduler" },
    { id: "p6-3", n: "P6.3", title: "The paged KV block manager", src: "course/P6-engine-internals/P6.3-paged-kv-block-manager" },
    { id: "p6-4", n: "P6.4", title: "Prefix caching with a radix tree", src: "course/P6-engine-internals/P6.4-prefix-cache-radix-tree" },
    { id: "p6-5", n: "P6.5", title: "Spec-decode verification and GPU sampling", src: "course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling" },
    { id: "p6-6", n: "P6.6", title: "CUDA graph capture", src: "course/P6-engine-internals/P6.6-cuda-graph-capture" },
    { id: "p6-7", n: "P6.7", title: "Project: a tiny GPU engine", src: "course/P6-engine-internals/P6.7-project-tiny-engine-gpu" } ] },
  { phase: "Cap", title: "Capstone", weeks: "41–44", blurb: "Survive a region failure, then publish a benchmark teardown anyone can reproduce.", lessons: [
    { id: "c1", n: "C1", title: "Multi-region failover", src: "course/capstone/C1-multi-region-failover" },
    { id: "c2", n: "C2", title: "The public benchmark teardown", src: "course/capstone/C2-public-benchmark-teardown" } ] },
  { phase: "Lane B", title: "CUDA on LeetGPU", weeks: "1–44", blurb: "The concepts behind each level of kernel problems, then the problems themselves.", lessons: [
    { id: "l1", n: "L1", title: "Launches and indexing", src: "laneB-cuda/L1-launch-and-indexing" },
    { id: "l2", n: "L2", title: "Memory and shared memory", src: "laneB-cuda/L2-memory-and-shared-memory" },
    { id: "l3", n: "L3", title: "Reductions and scans", src: "laneB-cuda/L3-reductions-and-scans" },
    { id: "l4", n: "L4", title: "Fused elementwise ops and norms", src: "laneB-cuda/L4-fused-elementwise-and-norms" },
    { id: "l5", n: "L5", title: "GEMM and tensor cores", src: "laneB-cuda/L5-gemm-and-tensor-cores" },
    { id: "l6", n: "L6", title: "Attention and Triton", src: "laneB-cuda/L6-attention-and-triton" } ] }
];
