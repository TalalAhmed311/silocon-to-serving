# Exercise 3 (hard): a fused min-p + top-k sampler (T2)

Write one Triton (or CUDA) kernel that, for each row: computes the max and the softmax normaliser (online, P5.5), applies min-p and top-k (top-k via a threshold search like top-p — count of elements ≥ τ is monotone too), and samples with a per-row Philox seed, in **one pass over the logits plus the threshold iterations**. Start from Lane B L4 #104 (min-p) and #60 (top-p).

Benchmark against `sample_torch` and a sort-based PyTorch sampler with `bench/sampler_bench.py` (add your kernel as a third column), V ∈ {32k, 128k}, B ∈ {1, 64, 256}, on the GPU. Check correctness with the same statistical test as `test_torch_sampler_matches_numpy_filters`. `TODO(run-on: g6.xlarge)`.
