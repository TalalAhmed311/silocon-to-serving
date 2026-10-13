# R.6 — HTTP and networking basics

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) any laptop, no GPU, no internet needed (tests run in-process) |
| **Time** | ≈30 min reading + ≈3 h hands-on |
| **Prerequisites** | A loop and a function in some language. [R.3](../R.3-terminal-git-build/README.md) (terminal) and [R.4](../R.4-python-and-numpy/README.md) (Python) help |
| **You will build** | 4 tested Python exercises: an SSE parser, TTFT/ITL metrics, a validated JSON API, a retry policy |

> Already comfortable with requests, status codes, streaming responses and retries? Do the quiz. If you get every question right, skip to [R.7](../R.7-cloud-account-and-cost-safety/README.md).

## Learning objectives

By the end of this module you can:

1. Explain client, server, IP address, port and socket, and read `localhost:8000`.
2. Say what DNS, TCP and TLS each cost before the first byte arrives, and tell latency from bandwidth.
3. Read and write an HTTP request and response, including an OpenAI-style `/v1/chat/completions` body.
4. Choose what a gateway should do with 200, 400, 401, 404, 429, 500 and 503.
5. Parse a server-sent-events stream and read TTFT and inter-token latency off it.
6. Decide when a retry is safe, and add backoff with jitter.

## Why this matters for inference engineering

[P1.3](../../P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/README.md) measures time to first token (TTFT) and inter-token latency (ITL) on a streaming connection. [P2.7](../../P2-serving-engines/P2.7-gateway-basics/README.md) builds a gateway that routes, rate-limits and retries requests. Both assume you know what travels over the wire and what the status codes mean. This module teaches exactly that.

---

## 1. Client, server, address, port

A **server** is a program that waits for requests. A **client** sends them. An **IP address** names a machine (`127.0.0.1` always means "this machine"; the name `localhost` points to it). A **port** is a number from 0 to 65535 that picks one program on that machine. `localhost:8000` is "the program listening on port 8000 here". The OS gives the client and server a **socket** each, the two ends of the connection.

## 2. TCP, latency and bandwidth

**TCP** gives you a reliable, ordered stream of bytes over an unreliable network. Opening one costs a round trip: the client sends SYN, the server answers SYN-ACK, the client sends ACK, and now data can flow. **Latency** is how long one trip takes (the round-trip time, RTT). **Bandwidth** is how many bytes per second fit through. A small message is limited by latency; a big file is limited by bandwidth.

## 3. DNS and TLS

Before the first byte of your request, a client to a named host usually pays for **DNS** (turn `api.example.com` into an IP address, typically one lookup round trip, often cached) and, for `https`, **TLS** (agree on keys and check the server's certificate, roughly one to two extra round trips depending on the TLS version). Reusing a connection ("keep-alive") pays these once instead of per request.

## 4. An HTTP request and response

HTTP is text on top of TCP. A request has a **method** (`GET`, `POST`), a **path**, **headers** and an optional **body**. A response has a **status code**, headers and a body. `curl -v` shows both:

```text
$ curl -v http://localhost:8000/v1/models
> GET /v1/models HTTP/1.1
> Host: localhost:8000
> Accept: */*
>
< HTTP/1.1 200 OK
< content-type: application/json
< content-length: 52
<
{"object":"list","data":[{"id":"mock-llama-8b"}]}
```

Lines starting `>` are what the client sent; lines starting `<` came back. (The body above is an illustration of the shape, not a captured output.)

## 5. JSON APIs and the chat-completions body

Most LLM servers speak JSON over HTTP in the OpenAI style. A request to `POST /v1/chat/completions`:

```json
{
  "model": "mock-llama-8b",
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 16,
  "stream": true
}
```

The headers carry `Content-Type: application/json` and usually `Authorization: Bearer <key>`. With `"stream": false` the reply is one JSON object when generation finishes. With `"stream": true` it is a stream of chunks.

## 6. Status codes that matter for serving

| Code | Meaning | What a gateway does |
|---|---|---|
| 200 | OK | pass it through |
| 400 | the request is malformed | return it to the caller, never retry |
| 401 | missing or wrong credentials | reject; never forward to a backend |
| 404 | unknown path or model | return it; check the route table |
| 429 | too many requests (rate limit) | tell the caller to back off; may retry elsewhere |
| 500 | the server failed | retry on another backend only if the request is safe to repeat |
| 503 | overloaded or draining | try another backend; shed load |

## 7. Streaming with server-sent events

A model produces tokens one at a time, so it is wasteful to make the user wait for all of them. With **server-sent events (SSE)** the server keeps the response open and sends chunks as they are ready. Each event is one or more `data:` lines followed by a **blank line**:

```text
data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

data: [DONE]

```

The client timestamps each chunk. **TTFT** is the time from sending the request to the first chunk. **ITL** is the gap between consecutive chunks. Without streaming, the user sees nothing until the last token, so TTFT and end-to-end time are the same number.

## 8. Timeouts, retries and idempotency

A request can fail in three ways: no answer (timeout), an error status, or an answer that got lost. A **timeout** bounds how long you wait. A **retry** sends the request again. Retrying is safe only for **idempotent** requests, where doing it twice equals doing it once (`GET`, or a `POST` that carries an idempotency key). A 429 or 503 usually means the server did not take the work on, so retrying is safe. A timeout or 500 may mean the work happened, so retry only if idempotent. Wait longer after each failure (**exponential backoff**) and add randomness (**jitter**) so many clients do not retry together.

## 9. Concurrency

A server with 1,000 open streaming connections cannot give each its own idle thread cheaply. It uses **async I/O** (one thread juggles many sockets, switching when one is ready) or a pool of **workers**. An LLM gateway holds a connection open for the whole generation, so connection count, not request rate, is its limit. P2 returns to this.

## 10. Hands-on tools

- `curl -v URL` shows headers. `curl -N URL` turns off buffering so you see a stream arrive live.
- `curl -s -X POST URL -H 'Content-Type: application/json' -d '{"text":"hi"}'` sends JSON.
- FastAPI's `TestClient` calls an app in-process with no network. The course's gateway tests use `s2s.gatewaytest.running_gateway` (a real localhost server) when a real socket matters.

---

## Exercises

| # | Folder | What you write |
|---|---|---|
| 1 | `exercises/01-parse-sse` | `parse_sse(raw)`: events from raw bytes, multi-line `data:`, comments, `[DONE]` |
| 2 | `exercises/02-ttft-itl` | `stream_metrics`: TTFT, mean and p90 ITL, end-to-end from timestamps |
| 3 | `exercises/03-tiny-api` | a FastAPI `POST /v1/echo` with pydantic validation and 400/422 errors |
| 4 | `exercises/04-retry-policy` | `should_retry` and exponential backoff with full jitter |

```bash
uv run pytest course/R-readiness/R.6-http-and-networking/exercises/01-parse-sse
S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.6-http-and-networking
```

## Go further

Links only; all UNVERIFIED from the build environment (see `SOURCES.md`).

- MDN, *An overview of HTTP*: https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview
- MDN, *Using server-sent events*: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events

## Go down when…

You are ready for [R.7 A safe cloud account](../R.7-cloud-account-and-cost-safety/README.md), and later [P1.3 Latency and throughput metrics](../../P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/README.md) and [P2.7 Gateway basics](../../P2-serving-engines/P2.7-gateway-basics/README.md).
