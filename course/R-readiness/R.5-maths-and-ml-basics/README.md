# R.5 — The maths and ML you need

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) any laptop, no GPU |
| **Time** | ≈40 min reading + ≈3 h hands-on |
| **Prerequisites** | School maths (multiplication, powers, fractions) and a loop and a function in some language. [R.4](../R.4-python-and-numpy/README.md) helps for the NumPy exercises |
| **You will build** | 4 tested Python exercises: matrix-vector product, stable softmax, a 2-layer MLP, next-token selection |

> Already comfortable with vectors, matrix products, softmax and "logits to next token"? Do the quiz. If you get every question right, skip to [R.6](../R.6-http-and-networking/README.md).

## Learning objectives

By the end of this module you can:

1. Compute a dot product and a matrix-vector product by hand, and count their FLOPs (2 per multiply-add).
2. Give the shape and cost of a matrix-matrix product: `(m, k) @ (k, n)` costs `2·m·k·n` FLOPs.
3. Explain what `exp` and `log` do, and write softmax in its numerically stable form.
4. Write `y = activation(W x + b)` and count the parameters of a small network.
5. Say what a logit is, trace text through tokens, ids and embeddings to a next token, and explain what temperature does.

## Why this matters for inference engineering

[P1.1](../../P1-inference-fundamentals/P1.1-transformer-forward-pass/README.md) walks one token through a Llama-style model and counts the weights, FLOPs and bytes of each step. It uses matrix-vector products, softmax and a gated MLP on almost every line. If those are familiar, P1.1 is a count. If not, it is a wall of symbols. This module is the shortest path from school maths to being able to read it.

---

## 1. Vectors and the dot product

A **vector** is a list of numbers. The **dot product** of two vectors multiplies them position by position and adds the results: `[1, 2, 3] · [4, 5, 6] = 4 + 10 + 18 = 32`. It is large when the two vectors point the same way, zero when they are perpendicular and negative when they point apart. Attention in a transformer is built from dot products: "how well does this query match each key?".

## 2. Matrix times vector

A **matrix** is a grid of numbers. `W @ x` with `W` of shape `(m, n)` and `x` of length `n` gives `m` numbers: row `i` of `W` dotted with `x`. Each multiply-add is 2 floating-point operations (FLOPs), so the cost is `2·m·n`. One `q_proj` in Llama-3-8B is 4096 × 4096: `2 × 4096 × 4096 = 33,554,432` FLOPs per token. This is why a model's cost is about `2 × parameters` FLOPs per token.

## 3. Matrix times matrix

`(m, k) @ (k, n) -> (m, n)`. The inner sizes must match. Each of the `m·n` outputs is a dot product of length `k`, so the cost is `2·m·k·n` FLOPs. Prefill (reading a whole prompt at once) is a matrix-matrix product; decode (one token at a time) is a matrix-vector product.

## 4. exp and log

`exp(x)` is `e^x` with `e ≈ 2.718`. It is always positive, and it grows fast: `exp(0) = 1`, `exp(1) ≈ 2.72`, `exp(2) ≈ 7.39`, `exp(3) ≈ 20.09`. That makes it a good way to turn any scores (including negative ones) into positive weights. `log` undoes `exp`, and `log(a·b) = log(a) + log(b)`: products become sums, which is why probabilities are often handled as logs.

## 5. Softmax, and the stable version

**Softmax** turns a vector of scores into probabilities: `p_i = exp(x_i) / Σ_j exp(x_j)`. For scores `[2, 1, 0]`: the exponentials are `[7.389, 2.718, 1]`, their sum is `11.107`, so the probabilities are `[0.665, 0.245, 0.090]`.

The formula breaks for big scores. A float32 tops out near `3.4e38`, and `exp(89)` is already larger, so `exp(1000)` becomes `inf` and the result is `nan`. The fix: subtract the largest score first. `softmax(x) == softmax(x - max(x))` because the shift cancels in the ratio, and now every exponent is `≤ 0`.

## 6. A neuron and a layer

A **layer** computes `y = activation(W x + b)`: multiply by a weight matrix, add a bias vector, then apply a simple curve to every element. **ReLU** is `max(0, x)`. **SiLU** is `x · sigmoid(x)`, a smooth version that dips slightly below zero for negative `x` (Llama's MLP uses it). Without the curve, stacked layers would collapse into one matrix.

## 7. Networks and parameters

A **network** is layers stacked: the output of one is the input of the next. The **parameters** are the numbers in all the `W` and `b`. For a 2-layer MLP with sizes 784 → 128 → 10: layer 1 has `128·784 + 128 = 100,480`, layer 2 has `10·128 + 10 = 1,290`, total `101,770`. Llama-3-8B is the same idea with about 8.03 billion parameters.

## 8. Training and inference

**Training** adjusts the parameters until the outputs are good. **Inference** only reads them and runs the forward pass. This course is about inference: the weights are fixed files, and the work is moving them and multiplying by them quickly.

## 9. From text to a next token

Text is split into **tokens** (pieces of words). Each token has an integer **id** in a fixed **vocabulary** (Llama-3-8B: 128,256 entries). The id selects a row of an **embedding** table, giving a vector of 4096 numbers. The layers transform those vectors. The last step turns one vector into 128,256 scores, one per vocabulary entry: the **logits**. Softmax turns logits into probabilities, and one token is chosen.

## 10. Sampling

**Greedy** takes the highest logit. **Sampling** draws a token at random in proportion to its probability. **Temperature** `T` divides the logits before softmax: `T < 1` sharpens (more greedy), `T > 1` flattens (more random), and `T → 0` approaches greedy. For logits `[2, 1, 0]`, `T = 1` gives `[0.665, 0.245, 0.090]`, and `T = 0.5` (logits become `[4, 2, 0]`) gives `[0.867, 0.117, 0.016]`. P0.5 builds a sampler; P2 shows what servers expose.

---

## Exercises

All four run with `uv run pytest`. Add `S2S_SOLUTIONS=1` to run the reference solutions and confirm the tests pass.

| # | Folder | What you write |
|---|---|---|
| 1 | `exercises/01-dot-and-matvec` | `dot`, `matvec` with loops, and `matvec_flops` |
| 2 | `exercises/02-softmax` | stable softmax over any axis; tests include `[1000, 1001]` |
| 3 | `exercises/03-mlp-forward` | SiLU, a 2-layer MLP forward pass, and `param_count` |
| 4 | `exercises/04-next-token` | greedy token, and top-k with probabilities and temperature |

```bash
uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/01-dot-and-matvec
S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.5-maths-and-ml-basics
```

## Go further

Links only; all UNVERIFIED from the build environment (see `SOURCES.md`).

- 3Blue1Brown, *Essence of linear algebra*: https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown, *Neural networks*: https://www.3blue1brown.com/topics/neural-networks
- Karpathy, *Neural Networks: Zero to Hero* (code from scratch): https://karpathy.ai/zero-to-hero.html
- NumPy, *the absolute basics for beginners*: https://numpy.org/doc/stable/user/absolute_beginners.html

## Go down when…

You are ready for [R.6 HTTP and networking basics](../R.6-http-and-networking/README.md), and then [P1.1 The transformer forward pass](../../P1-inference-fundamentals/P1.1-transformer-forward-pass/README.md).
