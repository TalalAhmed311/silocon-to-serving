# Exercise 4 (hard): fused cross-entropy (T2)

`loss[r] = logsumexp(logits[r]) − logits[r][target[r]]`. The naive way materializes softmax (a full write and re-read of `rows × vocab`) and then takes `−log p[target]`. Fused, it's **one read** of the logits:

- online `(m, d)` over the row, as in exercise 1
- the thread that sees column `target[r]` saves that logit (shared memory)
- `loss = m + log(d) − logit[target]`. Never compute `log(softmax)`: it underflows for confident predictions

The test uses vocabularies up to 128,256 (Llama 3's). `--bench` reports GB/s at 4096 × 128256 against half the copy bandwidth (read-only). This is why training frameworks fuse the LM head's loss: the logits tensor is the largest activation in the model.
