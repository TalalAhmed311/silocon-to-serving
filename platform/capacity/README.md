# capacity: D3, the model × GPU capacity calculator

Built in [P1.5](../../course/P1-inference-fundamentals/P1.5-project-capacity-calculator/README.md) and used from then on:

- P2: predict a run before you start it
- P3: pod sizing
- P4: choosing a TP degree
- #2: dollars per token

```bash
PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4
PYTHONPATH=platform uv run python -m capacity.cli --model llama3-70b --gpu H100-SXM --tp 4 --weights fp8 --kv fp8 --context 8192
PYTHONPATH=platform uv run python -m capacity.cli --model path/to/config.json --gpu L40S
```

**Every output is a *prediction*.** It is computed from:

- the model's config (presets are UNVERIFIED, see `presets/README.md`)
- `gpu_specs.yaml` (UNVERIFIED until P1.4 exercise 1 is done)
- three named assumptions, printed with every result: `mem_util`, `bw_util` and `mfu`

P2.1 and P2.3 measure the real numbers on a GPU and compare them with these predictions.
