# Exercise 2: top-p without a sort

Implement `top_p_threshold` by bisection (README §1). Tests: `test_top_p_same_set_as_sort` (10 heavy-tailed Dirichlet distributions × 4 values of top_p must keep **exactly** the sort reference's set) and `test_top_p_peaked`.

With the `[torch]` extra, `test_torch_sampler_matches_numpy_filters` checks the batched PyTorch sampler (`sample_torch`) samples from the same filtered distribution as the NumPy reference.

**Then:** how many bisection iterations do you need so the result is exact in fp32 for V = 128k? What's a failure case for bisection (think: ties at the threshold), and how does the sort-based version behave on the same input?
