# Exercise 4: from logits to a next token

The model's last step gives one score (a **logit**) for every token in the vocabulary. Implement `greedy` (take the highest) and `top_k` (the k most likely tokens with their probabilities, after dividing the logits by a temperature) in `nexttoken.py`. Probabilities are taken over the whole vocabulary, not renormalised over the k you return. Low temperature sharpens the distribution; high temperature flattens it.

```bash
uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/04-next-token
```
