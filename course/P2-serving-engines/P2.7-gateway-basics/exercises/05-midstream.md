# Exercise 5 — SSE passthrough with mid-stream failure (hard)

Run a mock with `MOCKLLM_FAIL_MIDSTREAM_RATE=1.0`: every stream is cut after 3 tokens. Behind #6, the client must receive the 3 tokens, then **one** explicit error event (`"type": "upstream_interrupted"`), and never a 4th token from a silently retried backend.

1. Read `Gateway._stream` in `platform/gateway/app.py` and find the line that makes retries impossible after the first byte.
2. **Test (`test_midstream.py`):** it checks exactly that behaviour, plus that a stream that fails *before* the first byte is transparently retried on the healthy backend.
3. **Design question (written):** a client wants resumable streams. Propose a protocol in which the gateway may retry mid-stream *safely*. Hint: the client sends back the tokens it has already received as a prefix, and the backend must support prompt continuation, or the gateway does it with prefix caching. What breaks with sampling at temperature > 0?
