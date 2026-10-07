# Exercise 3 — Trace propagation across gateway → backend (T0)

`test_tracing.py` runs the gateway with `S2S_TRACING=memory` in front of two mock backends, one of which always fails. It sends one non-streaming request and inspects the finished spans:

- a single trace ID across all of them
- `gateway.request` is the parent of both `gateway.backend_attempt` spans: a failed one (503), then a successful one (200)
- the backend request carried a `traceparent` header, checked with a tiny capture server

**Then extend it:** the streaming path (`Gateway._stream`) creates no spans yet. Add a `gateway.stream_attempt` span per attempt, with attributes `s2s.backend`, `s2s.ttft_s` and `s2s.completion_tokens`, plus an event on mid-stream interruption. Remove the `pytest.skip` from `test_stream_spans`.
