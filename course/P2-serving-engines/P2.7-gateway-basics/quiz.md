# P2.7 quiz

<details><summary><b>1.</b> Why can't a gateway retry a streaming request after the first byte?</summary>

The client has already received output. A second backend's output would duplicate or splice tokens, and sampling isn't deterministic.
</details>

<details><summary><b>2.</b> What is a retry budget, and what failure does it prevent?</summary>

A cap on retries as a fraction of requests. It prevents retry storms, where a partial outage multiplies load on the remaining healthy backends.
</details>

<details><summary><b>3.</b> Why least-outstanding-requests over round-robin for LLM backends?</summary>

Request costs vary by orders of magnitude. In-flight count tracks real load, while round-robin assumes equal cost.
</details>

<details><summary><b>4.</b> Why check an <i>estimate</i> but bill the <i>actual</i> usage?</summary>

The actual output length is unknown until the response finishes. Billing from the backend's `usage` is exact, and the estimate only gates admission.
</details>

<details><summary><b>5.</b> Why store tenant keys as hashes?</summary>

A leaked config or log then doesn't leak usable credentials. Compare hashes in constant time to avoid timing side channels.
</details>
