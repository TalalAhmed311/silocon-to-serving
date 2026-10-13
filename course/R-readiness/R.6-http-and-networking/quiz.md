# R.6 quiz

<details><summary><b>1.</b> What does <code>localhost:8000</code> mean?</summary>

"The program listening on port 8000 on this same machine." `localhost` resolves to `127.0.0.1`, an address that never leaves your computer. The port number picks one program among many.
</details>

<details><summary><b>2.</b> Why does a small request to a far-away server feel slow even on a very fast link?</summary>

Small messages are limited by **latency** (round trips), not bandwidth. TCP setup is one round trip, TLS adds one or two, and the request and response take one more. With a 100 ms RTT that is several hundred milliseconds before the first byte, whatever the link speed.
</details>

<details><summary><b>3.</b> A client uses a fresh connection for every request to <code>https://api.example.com</code>. Which costs does keep-alive remove?</summary>

The TCP handshake and the TLS handshake (and often the DNS lookup is cached anyway). Reusing one connection pays them once and then sends many requests over it.
</details>

<details><summary><b>4.</b> Name the three parts of an HTTP request and the three parts of a response.</summary>

Request: method + path (the request line), headers, optional body. Response: status code (the status line), headers, optional body.
</details>

<details><summary><b>5.</b> A client sends <code>{"model": "x", "messages": "hi"}</code> to a chat endpoint that expects a list of messages. Which status should it get: 400, 422, 401 or 503?</summary>

**422** if the server uses schema validation as FastAPI does (the JSON parses but has the wrong shape), or **400** on servers that treat all bad input as 400. Not 401 (credentials are fine) and not 503 (the server is healthy). The point: the caller's request is wrong, so a retry cannot help.
</details>

<details><summary><b>6.</b> A stream arrives as chunks at t = 0.50, 0.55, 0.60, 0.80 s after the request was sent. What are TTFT and the ITL values?</summary>

TTFT = **0.50 s**. The gaps are 0.05, 0.05 and 0.20 s, so the mean ITL is 0.10 s. The 0.20 s gap is a stall the mean hides, which is why P1.3 uses percentiles.
</details>

<details><summary><b>7.</b> Why is it safe to retry a 429 for a non-idempotent request, but not a timeout?</summary>

A 429 is an answer: the server refused before doing the work. A timeout is silence: the server may have finished the work and the reply got lost, so a retry could do it twice. For a timeout, retry only if the request is idempotent (or carries an idempotency key).
</details>

<details><summary><b>8.</b> Why add random jitter to exponential backoff?</summary>

Without jitter, every client that failed at the same moment waits the same time and retries at the same moment, hitting the server in a synchronised burst. A random wait between 0 and the backoff bound spreads the retries out.
</details>
