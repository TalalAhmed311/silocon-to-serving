# 112: Attention with Sinks (L6, practice)

Problem: [LeetGPU #112](../../leetgpu-map.md). A learned per-head "sink" logit joins the softmax denominator but contributes no value.

**Hint ladder**

1. `out = Σ e^{a_j} v_j / (e^{s} + Σ e^{a_j})`: the sink only increases the normalizer.
2. In online-softmax terms, start the state at `(m, l, o) = (s, 1, 0)` instead of `(−∞, 0, 0)`. Everything else is unchanged.
3. Don't confuse this with StreamingLLM's "attention sinks" (always keep the first few tokens in a sliding window), which is a **mask**. Read the statement.

**Solution outline:** the shared core with `sinks`.
