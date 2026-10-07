# Exercise 2: where does a step's time go? (T2)

On a one-GPU instance (see [aws.md](../aws.md)), clone nano-vllm at the pinned SHA and add wall-clock timers (with `torch.cuda.synchronize()` around GPU work) to `LLMEngine.step` for: scheduling, input preparation (`prepare_prefill`/`prepare_decode`), the forward pass, sampling, and post-processing.

Run its example with a small model (Qwen3-0.6B, which nano-vllm's README uses; check the licence) at batch 1, 8 and 64 decode-heavy requests and fill in:

| batch | schedule (ms) | prepare (ms) | forward (ms) | sample (ms) | postprocess (ms) | CPU share of step |
|---|---|---|---|---|---|---|
| 1 | `TODO(run-on: g6.xlarge)` | | | | | |
| 8 | | | | | | |
| 64 | | | | | | |

Then answer: at which batch size does CPU-side work stop mattering? Turn CUDA graphs off (`enforce_eager=True`) and re-measure batch 1: how much of the forward pass was launch overhead? (P6.6 explains the result.)
