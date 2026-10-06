# bakeoff: #7, the quantized serving bakeoff

The same base model in several precisions, on one GPU, measured three ways: **quality** (lm-evaluation-harness), **speed** (#4 loadgen: TTFT, decode tok/s, knee) and **memory** (VRAM, KV-cache tokens).

```bash
python platform/bakeoff/quantize.py --base /opt/models/base --scheme fp8-dynamic --out /opt/models/base-fp8   # T2
python platform/bakeoff/run.py platform/bakeoff/variants.yaml                                                 # T2
python platform/bakeoff/table.py results/bakeoff                                                              # T0
```

Built in [P2.5](../../course/P2-serving-engines/P2.5-quantization/README.md). Tool versions are pinned in `env/requirements-quant.txt`: LLM Compressor 0.14.0, GPTQModel 7.5.0, lm-eval 0.4.13. AutoAWQ is deprecated, and AWQ now comes from LLM Compressor.
