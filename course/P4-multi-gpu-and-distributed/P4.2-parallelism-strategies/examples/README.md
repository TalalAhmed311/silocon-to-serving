# P4.2 examples

| File | Tier | What |
|---|---|---|
| [`01_tp_mlp_numpy.py`](01_tp_mlp_numpy.py) | T0 | Megatron column→row split of a SwiGLU MLP on simulated ranks; error vs unsplit; the wrong split order |
| [`02_tp_torch_gloo.py`](02_tp_torch_gloo.py) | T0 | the same with real processes and one `all_reduce` |
| [`03_vllm_tp.sh`](03_vllm_tp.sh) | T3 | vLLM TP = 1/2/4 on a 4-GPU box, with #4 runs for each |
| [`04_fsdp2_toy.py`](04_fsdp2_toy.py) | T3 | FSDP2 `fully_shard` vs DDP: parameters and memory per rank |
| [`../parallel.py`](../parallel.py) | T0 | reference TP MLP, pipeline simulator (GPipe and 1F1B), ZeRO memory accounting |
