# R.5 quiz

<details><summary><b>1.</b> What is <code>[1, 2, 3] · [4, 5, 6]</code>, and what does a large positive dot product tell you?</summary>

**32** (`4 + 10 + 18`). A large positive value means the vectors point in a similar direction. Zero means perpendicular; negative means they point apart.
</details>

<details><summary><b>2.</b> How many FLOPs does a (4096 × 4096) matrix times a 4096-vector cost, and how many bytes does the matrix take in bf16?</summary>

`2 × 4096 × 4096 = 33,554,432` FLOPs (about 33.6 million). The matrix has 16,777,216 numbers; at 2 bytes each that is 33,554,432 bytes = 32 MiB. Notice the two numbers are equal: 2 FLOPs per weight and 2 bytes per weight, so each byte read buys one FLOP. That ratio matters from P1.1 onward.
</details>

<details><summary><b>3.</b> What shape does <code>(8, 3) @ (3, 5)</code> have, and how many FLOPs?</summary>

Shape **(8, 5)**. The inner 3 must match and disappears. Cost `2 × 8 × 3 × 5 = 240` FLOPs.
</details>

<details><summary><b>4.</b> Compute softmax of <code>[0, 0]</code> and of <code>[0, ln 3]</code> (where <code>ln 3 ≈ 1.099</code>).</summary>

`[0.5, 0.5]` for equal scores. For `[0, ln 3]` the exponentials are `[1, 3]`, the sum is 4, so the result is `[0.25, 0.75]`. A score gap of `ln 3` makes one option 3 times as likely.
</details>

<details><summary><b>5.</b> Why does softmax of <code>[1000, 1001]</code> give <code>nan</code> in float32 with the textbook formula, and what is the fix?</summary>

`exp(1000)` overflows float32 (the limit is about 3.4e38; `exp(89)` already passes it), giving `inf`, and `inf / inf` is `nan`. Subtract the maximum first: `[−1, 0]`, whose exponentials are `[0.368, 1]`, giving `[0.269, 0.731]`. The shift cancels in the ratio, so the answer is the same.
</details>

<details><summary><b>6.</b> How many parameters does an MLP with sizes 3 → 4 → 2 have (with biases)?</summary>

Layer 1: `4·3 + 4 = 16`. Layer 2: `2·4 + 2 = 10`. Total **26**. Parameters are the numbers in every W and b; the input and the activations are not parameters.
</details>

<details><summary><b>7.</b> What is the difference between training and inference, and which does this course cover?</summary>

Training changes the weights to reduce error (it needs the forward pass, a backward pass and an optimiser). Inference only runs the forward pass with fixed weights. The course covers **inference**: serving a trained model fast and cheaply.
</details>

<details><summary><b>8.</b> Logits are <code>[2, 1, 0]</code>. What happens to the top probability when temperature goes from 1 to 0.5?</summary>

At T = 1 the probabilities are `[0.665, 0.245, 0.090]`. At T = 0.5 the logits become `[4, 2, 0]`, giving `[0.867, 0.117, 0.016]`. The top probability rises from 0.665 to 0.867: lower temperature sharpens the distribution towards greedy.
</details>
