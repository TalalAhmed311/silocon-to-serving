# specdec: #10, the speculative decoding prototype

- **`core.py`**: exact speculative sampling (accept with min(1, p/q), resample from the normalized max(0, p − q), plus a bonus token), with acceptance accounting and the closed-form expected speedup.
- **`hf_models.py`**: Hugging Face models as draft and target. The target scores all k+1 positions in one forward pass.
- **`bench.py`**: acceptance rate, tokens per target call and wall-clock speedup vs target-only decoding.

Built in [P2.6](../../course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/README.md). In P6.5 the verification moves onto the GPU.
