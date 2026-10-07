# Exercise 1: distribution preservation of batched verification

**T0.** Implement `verify_batch` in `my_sampling.py`. `test_verify_preserves_first_token_distribution` draws 5,000 drafts from q, verifies them against p, and checks that the first emitted token is distributed as p₀ (tolerance 0.03 — about 4σ at this sample size for probabilities near 0.2). `test_verify_lengths_and_prefix` checks the output shape.

**T2.** Port it to PyTorch on the GPU, batched over B with no Python loop over positions: compute the accept mask `[B, k]`, find each row's first rejection with `cumprod` (or a prefix scan), and sample all residuals in one `torch.multinomial` call. Then extend the statistical test to **every** position's marginal, conditioned on the prefix being accepted (the theorem says each emitted token is distributed as the target given its prefix). Compare with Lane B L6 #87's CUDA kernel.

**Then:** why is the test on the *first* token enough to catch the most common bugs (forgetting the clamp, sampling the residual from p instead of p − q)?
