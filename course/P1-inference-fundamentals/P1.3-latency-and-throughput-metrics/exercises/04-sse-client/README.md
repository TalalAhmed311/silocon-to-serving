# Exercise 4 — An SSE client that measures TTFT itself (hard)

Implement `async stream_completion(client, base_url, prompt, max_tokens) -> dict` in `client.py`. It sends `POST /v1/completions` with `"stream": true`, parses the Server-Sent Events (`data: {...}` lines, ending with `data: [DONE]`), and returns:

`{"text", "ttft", "itl": [...], "n_tokens", "usage"}`

Measure from **just before sending** the request to **the first chunk carrying non-empty text**. Read `usage` from the final chunk.

**Test:** the test starts `platform/mockllm` on a localhost port in a background thread, with a fast `TIME_SCALE`, so nothing leaves the machine. Then it checks:

1. the token count equals `max_tokens`
2. `usage.completion_tokens` matches
3. client TTFT ≥ the server-observed TTFT for the same single request (scraped from `/metrics` as the histogram sum)
4. ITLs are positive and their sum ≈ E2E − TTFT

**Why (3) holds:** the client's clock starts before the HTTP request is even written, and stops after the bytes cross the socket. The server's clock covers only its own part. In production, the difference between the two is your network and proxy overhead, a metric worth having on a dashboard (P3.5).
