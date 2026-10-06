# Exercise 1 — Parameter count formula (easy)

Implement `count_params(cfg: dict) -> int` in `params.py`, using only the formula from §2 of the lesson, with no tensors. `cfg` uses the HF `config.json` keys. Handle `tie_word_embeddings`.

**Test:** compared against the exact sum of tensor sizes for 4 random configs (`conftest.weight_shapes`). If the `torch` extra is installed, it is also compared against `sum(p.numel())` of a real `LlamaForCausalLM(LlamaConfig(**cfg))`.

**Then, by hand:** load the real Llama-3-8B config from the Hub (`transformers.AutoConfig.from_pretrained(...)`; the model is gated, so accept its license first) or from a local snapshot. Check the lesson's table, and fix `03_flop_counter.py`'s preset if anything differs. Record the result in your notes. This is the "verify before you cite" step.
