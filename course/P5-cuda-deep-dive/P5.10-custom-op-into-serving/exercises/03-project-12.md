# Exercise 3: #12 end to end (T2)

Follow [aws.md](../aws.md):

1. **Micro:** `bench_op.py`, filling in the first README bench table.
2. **Same output:** start vLLM with and without `S2S_RMSNORM=1` (greedy sampling, `temperature=0`) and send the same 20 prompts. The generated tokens should be identical, or differ only where bf16 rounding ties go the other way. Explain any difference.
3. **#4:** three runs each of `loadgen.cli` (same seed, same rates) with the flag off and on. Fill in the second table **with the spread**, not just one number.
4. **#13 + nsys:** one trace per configuration. Confirm by kernel name that your kernel ran, and report its share of a decode step.
5. Conclusion in 5–8 sentences: is the change worth shipping? On what evidence?
