# P6.5 quiz

<details><summary><b>1.</b> Why can the top-p cut-off be found by bisection?</summary>

The kept mass, Σ p_i over p_i ≥ τ, is monotone non-increasing in τ. The cut-off is the largest τ whose kept mass is still ≥ top_p, so bisection on τ converges to it with one parallel reduction per iteration.
</details>

<details><summary><b>2.</b> In which order are temperature, top-k, min-p and top-p applied, and why does temperature come first?</summary>

Temperature first (it reshapes the distribution every other filter looks at), then the truncations. With temperature applied last, top-p and min-p would select a different set than the user asked for.
</details>

<details><summary><b>3.</b> How do you keep a request's samples reproducible when its batch-mates change every step?</summary>

Give each request its own seeded generator and consume it once per token, independently of the batch.
</details>

<details><summary><b>4.</b> A draft token has p = 0.1, q = 0.4. With what probability is it accepted, and on rejection what do you sample from?</summary>

0.25. On rejection, sample from max(p − q, 0) renormalised, at that position, then stop.
</details>

<details><summary><b>5.</b> All k drafts are accepted. How many tokens does the step emit?</summary>

k + 1: the k accepted drafts plus a bonus token sampled from the target's distribution at position k + 1, which the same forward pass already computed.
</details>
