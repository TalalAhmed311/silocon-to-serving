# Exercise 1: register the op and pass `opcheck` (T0 → T2)

1. Read `platform/kernels/torch_ext/ops.py`. Explain each decorator: `custom_op(…, mutates_args=…)`, `register_kernel("cuda")`, `register_fake`.
2. `uv run --extra torch pytest platform/kernels/tests_py -m torch -k "semantics or opcheck"`. Both pass on CPU (T0).
3. **Break it on purpose**, once each, and read opcheck's error: remove `"x"` from `mutates_args`; make the fake return `torch.empty_like(x)`; make the reference write into a new tensor instead of in place.
4. On the GPU box: `test_cuda_impls` for both implementations (fp32, bf16, fp16). The residual must match **exactly**. Why can it?
