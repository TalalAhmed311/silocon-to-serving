# s2s-vllm-plugin: #12, swap in our kernel behind a flag

```bash
uv pip install -r env/requirements-serving.txt && uv pip install -e platform/kernels/vllm_plugin
S2S_RMSNORM=1 S2S_RMSNORM_IMPL=cuda vllm serve <model> …     # look for "s2s plugin: fused_add_rms_norm → …" in the log
```

The plugin is loaded in every vLLM process. It's a no-op unless `S2S_RMSNORM=1`. See `s2s_vllm_plugin/__init__.py` for the exact call path it replaces, and [P5.10](../../../course/P5-cuda-deep-dive/P5.10-custom-op-into-serving/README.md) for the end-to-end measurement.
