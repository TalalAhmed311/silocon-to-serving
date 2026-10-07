# P5.10: Wiring a custom op into PyTorch/vLLM (+ #12)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the op registration, `opcheck` and `torch.compile` tests (CPU reference implementation). ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for the CUDA/Triton implementations and the end-to-end #12 bench on vLLM (`g6.xlarge`) |
| **Time** | ≈25 min reading + ≈10 h hands-on |
| **Prerequisites** | P5.5 (fused residual RMSNorm), P5.9 (Triton), P2.1 (vLLM), P2.4 (#4 load test), P3.5 (#13 tracing) |
| **Reading** | PyTorch docs: *Custom C++ and CUDA operators*, `torch.library` (UNVERIFIED for your version) · vLLM at the pinned SHA: `csrc/layernorm_kernels.cu`, `vllm/_custom_ops.py`, `vllm/model_executor/layers/layernorm.py`, `docs/design/plugin_system.md` |
| **You will build** | **#12**: your fused residual-add + RMSNorm running inside vLLM behind a flag, measured end to end |

## Learning objectives

1. Register a custom operator with `torch.library.custom_op`, declaring which inputs it **mutates**, with a fake (meta) implementation.
2. Pass `torch.library.opcheck`, and make the op `torch.compile(fullgraph=True)`-safe.
3. Provide device-specific implementations: a JIT-built CUDA extension and a Triton kernel.
4. Find where vLLM dispatches a kernel and swap it via a plugin, with no fork needed for the experiment.
5. Measure the change **end to end** (#4 latency/throughput, #13 traces), not just in a microbenchmark.

---

## 1. Why "register" instead of "call"

vLLM runs the model under `torch.compile` and CUDA-graph capture (P6.6). Both need every op to have a **schema** (types, which args are mutated) and a **fake implementation** that propagates shapes without running. A plain Python function would be traced *into*, and a raw ctypes call would break capture. `torch.library.custom_op` gives you both, plus `opcheck`, which tests the registration itself: schema vs actual mutation, fake vs real output metadata, and the autograd/aot paths.

`platform/kernels/torch_ext/ops.py` registers `s2s::fused_add_rms_norm(x, residual, weight, eps) -> None`:

- **default implementation** (any device): the PyTorch reference, which is what CPU `opcheck` runs (T0)
- **`cuda` kernel**: `S2S_RMSNORM_IMPL=cuda` → `csrc/s2s_ext.cu` (JIT-built with `torch.utils.cpp_extension.load`), or `=triton` → `triton_kernels/rmsnorm.py`
- **fake**: returns None (the op only mutates)

Its semantics match vLLM's: `residual ← x + residual`, `x ← rmsnorm(residual)·w`, both **in place**.

## 2. Where vLLM calls it

In the pinned vLLM, the `RMSNorm` layer's forward calls `ops.fused_add_rms_norm(x, residual, weight, eps)` through `vllm._custom_ops`, which forwards to `torch.ops._C.fused_add_rms_norm` (compiled from `csrc/layernorm_kernels.cu`). Because the layer looks up `ops.fused_add_rms_norm` **at call time**, replacing that module attribute redirects every call. That's what `platform/kernels/vllm_plugin` does, from a `vllm.general_plugins` entry point, so it runs in every vLLM process (engine core and workers), and only when `S2S_RMSNORM=1`. Re-verify these paths after any vLLM upgrade.

## 3. Measuring #12

1. **Micro:** `torch_ext/bench_op.py` at serving shapes (1, 32, 256, 4096 tokens × 4096 hidden): ours (CUDA, Triton) vs vLLM's kernel vs eager.
2. **End to end:** the same vLLM server and #4 load (same seed) with `S2S_RMSNORM=0` vs `1`. Compare the knee, p90 TTFT and ITL. Expect a **small** effect: RMSNorm is a few percent of a decode step. That is the lesson. A 2× faster kernel on a 3% path is a ~1.5% win, and #4's run-to-run noise may be larger. Repeat runs and report the spread.
3. **Trace:** with #13's tracing on, confirm the requests went through the patched path (vLLM log line plus nsys kernel names: your kernel replaces `fused_add_rms_norm_kernel`).

> **Predict first.** From an nsys trace of one decode step (P5.3), what fraction of GPU time is `fused_add_rms_norm`? If your kernel is X% faster, what end-to-end ITL change do you expect? Write it down, then measure.

---

## Walkthrough

```bash
uv run --extra torch pytest platform/kernels/tests_py -m torch          # T0: CPU opcheck + torch.compile
```

The T2 path is in [aws.md](aws.md).

## What you should see

- T0: `opcheck` passes, and `torch.compile(fullgraph=True)` compiles and matches eager.
- T2: CUDA and Triton implementations match the reference (the residual exactly). Microbench GB/s close to vLLM's kernel.
- vLLM with the plugin: the log shows the swap, outputs are unchanged (same tokens with greedy sampling), and end-to-end deltas are within noise or a few percent.

`TODO(run-on: g6.xlarge)`

## Bench (#12)

| tokens | eager µs | vLLM kernel µs | ours (CUDA) µs | ours (Triton) µs |
|---|---|---|---|---|
| 1 / 32 / 256 / 4096 | | | | `TODO(run-on: g6.xlarge)` |

| run (same seed, #4) | knee req/s | p90 TTFT | p50 ITL |
|---|---|---|---|
| `S2S_RMSNORM=0` (×3) | | | `TODO(run-on: g6.xlarge)` |
| `S2S_RMSNORM=1` (×3) | | | |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Register the op and pass `torch.library.opcheck`](exercises/01-opcheck.md) | `tests_py/test_custom_op.py::test_opcheck_cpu` (+ GPU variants) |
| 2 | [Make it `torch.compile`-safe](exercises/02-compile.md) | `test_torch_compile_fullgraph_cpu` |
| 3 | [#12 end-to-end: micro + #4 + #13](exercises/03-project-12.md) | your two bench tables |
| 4 | *(hard)* [An upstream-quality PR description](exercises/04-pr.md) | the write-up (not submitted unless you choose to) |

## Common mistakes

- Not declaring mutation (`mutates_args`). The compiler then reorders or removes your in-place op.
- A fake implementation that allocates or returns the wrong metadata. opcheck catches it.
- Patching vLLM in the API-server process only, while the model runs in the engine-core process. Use the plugin entry point.
- Celebrating a microbenchmark win that doesn't show up end to end.

## Go deeper

- PyTorch `torch.library` docs and the "custom ops landing page" (UNVERIFIED URLs; search the docs for `custom_op`).
- vLLM `csrc/` and `vllm/_custom_ops.py` at the pinned SHA. SGLang `python/sglang/kernels/` for a second engine's approach.

**Next:** [P6 Engine internals](../../P6-engine-internals/syllabus.md).
