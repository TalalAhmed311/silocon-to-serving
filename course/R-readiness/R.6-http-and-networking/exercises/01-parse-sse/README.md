# Exercise 1: parse server-sent events

An LLM server streams its answer as `data: ...` lines, each event ended by a blank line, and finishes with `data: [DONE]`. Implement `parse_sse(raw: bytes)` in `sse.py` so it turns the raw bytes into a list of `{"event", "data"}` dicts. The docstring lists the rules: multi-line `data:`, comments, event names, CRLF line ends, `[DONE]`, and an unfinished last event. Read the MDN page on server-sent events (linked in the module README) if the format is new.

```bash
uv run pytest course/R-readiness/R.6-http-and-networking/exercises/01-parse-sse
```
