# P5.10 on AWS

Instance and safety as in [P2.1's aws.md](../../P2-serving-engines/P2.1-first-serve-with-vllm/aws.md) (vLLM on a **`g6.xlarge`**, `127.0.0.1` + API key), plus the [shared P5 setup](../aws-common.md) for the CUDA toolchain. This module needs ≈ 3 instance-hours.

```bash
uv sync --extra torch && uv pip install -r env/requirements-serving.txt && uv pip install -e platform/kernels/vllm_plugin
uv run pytest platform/kernels/tests_py -m "torch"                                # incl. GPU variants (JIT-builds the extension)
PYTHONPATH=platform:platform/kernels uv run python platform/kernels/torch_ext/bench_op.py | tee results/p5.10-micro.md
# end to end: same model, same flags; three #4 runs each
for flag in 0 1; do
  S2S_RMSNORM=$flag vllm serve /opt/dlami/nvme/m --host 127.0.0.1 --api-key "$S2S_API_KEY" --served-model-name m \
    > results/p5.10-vllm-$flag.log 2>&1 &
  until curl -sf -H "Authorization: Bearer $S2S_API_KEY" localhost:8000/v1/models >/dev/null; do sleep 5; done
  for run in 1 2 3; do PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 4 8 16 \
    --duration 90 --seed 0 --out results/p5.10-flag$flag-run$run.json; done
  kill %1; wait
done
grep "s2s plugin" results/p5.10-vllm-1.log
```

Teardown and auto-stop: see the shared P5 page.
