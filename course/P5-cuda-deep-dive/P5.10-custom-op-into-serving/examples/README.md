# P5.10 examples

| Path | What |
|---|---|
| [`platform/kernels/torch_ext/ops.py`](../../../../platform/kernels/torch_ext/ops.py) | the `s2s::fused_add_rms_norm` custom op: reference, CUDA and Triton implementations, fake |
| [`platform/kernels/torch_ext/csrc/s2s_ext.cu`](../../../../platform/kernels/torch_ext/csrc/s2s_ext.cu) | the CUDA extension (JIT-built) |
| [`platform/kernels/torch_ext/bench_op.py`](../../../../platform/kernels/torch_ext/bench_op.py) | microbenchmark vs vLLM's kernel and eager |
| [`platform/kernels/vllm_plugin/`](../../../../platform/kernels/vllm_plugin/README.md) | the vLLM plugin that swaps it in behind `S2S_RMSNORM=1` |
| [`platform/kernels/tests_py/test_custom_op.py`](../../../../platform/kernels/tests_py/test_custom_op.py) | opcheck, torch.compile and GPU correctness tests |
