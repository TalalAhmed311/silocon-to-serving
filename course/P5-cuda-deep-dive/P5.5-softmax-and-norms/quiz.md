# P5.5 quiz

<details><summary><b>1.</b> Merge (m₁, d₁) = (2, 3) with (m₂, d₂) = (5, 1).</summary>

M = 5, d = 3·e^{2−5} + 1·e^0 = 3e^{−3} + 1 ≈ 1.149.
</details>

<details><summary><b>2.</b> Why is softmax measured in GB/s rather than FLOP/s?</summary>

It does a few FLOPs per element, so it's bandwidth-bound. Its ceiling is the copy kernel.
</details>

<details><summary><b>3.</b> Fused add + RMSNorm vs separate kernels: how many row-passes each?</summary>

Separate: add reads x and r and writes r (3), then norm reads r and writes y (2) = 5. Fused: read x and r, write r and y = 4. Plus one launch fewer.
</details>

<details><summary><b>4.</b> Why is rtol ≈ 1e-2 justified for a bf16 output?</summary>

bf16 has 8 significant bits, so one rounding has relative error up to 2⁻⁸ ≈ 3.9e-3. With fp32 math in between, the output rounding dominates, and ~2.5u covers it.
</details>

<details><summary><b>5.</b> Why compute cross-entropy as logsumexp − logit[target] instead of −log(softmax[target])?</summary>

It's one read with no materialized probabilities, and it avoids log(0) when the softmax underflows for confident predictions.
</details>
