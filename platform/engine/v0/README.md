# #0 v0 — Tiny Inference Engine (CPU)

A Llama-architecture decoder in about 400 lines of C++17/20. It has no dependencies beyond the standard library and pthreads.

- Weights are **mmap'd zero-copy** from a safetensors file ([P0.2](../../../course/P0-systems-primer/P0.2-virtual-memory-and-mmap/README.md)).
- Every matvec is **SIMD + multithreaded** ([P0.3](../../../course/P0-systems-primer/P0.3-threads-atomics-caches/README.md), [P0.4](../../../course/P0-systems-primer/P0.4-simd-and-roofline/README.md)).
- It keeps a **KV cache**.
- It samples with **temperature + top-p**.

The lesson that teaches it is [P0.5](../../../course/P0-systems-primer/P0.5-project-tiny-engine-cpu/README.md). In P6.7 it is ported to CUDA as #0 v1.

```
src/
  config.hpp            reads HF config.json
  safetensors_view.hpp  zero-copy loader (from P0.2)
  thread_pool.hpp       parallel_for (from P0.3)
  simd.hpp              AVX2 / NEON / scalar (from P0.4)
  ops.hpp               matvec, rmsnorm, rope, attention, softmax, silu, sampling
  engine.hpp            weights + KV cache + forward(token, pos)
  main.cpp              CLI
reference/llama_numpy.py  the reference forward pass every implementation is tested against
tools/make_tiny_llama.py  deterministic random tiny model (CI, tests)
tools/convert_hf_model.py any HF LlamaForCausalLM → fp32 safetensors + tokenizer
tools/chat.py             text in/out around the binary
tests/test_engine.py      teacher-forced logits + greedy tokens vs NumPy
```

## Build and run (T0)

```bash
cmake -S platform/engine/v0 -B build/engine-v0 -DCMAKE_BUILD_TYPE=Release && cmake --build build/engine-v0 -j
uv run python platform/engine/v0/tools/make_tiny_llama.py build/tiny-llama
./build/engine-v0/s2s-engine --model build/tiny-llama --prompt-ids "1 2 3 4" --steps 32
uv run pytest platform/engine/v0/tests
```

Real text: `uv run --extra torch python platform/engine/v0/tools/convert_hf_model.py <hf-llama-model> build/my-model`, then `tools/chat.py --model build/my-model "Once upon a time"`.

## Limits (on purpose)

- fp32 weights only. Int8 weights are P0.5 exercise 5.
- One sequence at a time.
- The prompt is fed token by token, with no batched prefill.
- No RoPE scaling (`rope_scaling` in config.json).

Each of these is the subject of a later module.
