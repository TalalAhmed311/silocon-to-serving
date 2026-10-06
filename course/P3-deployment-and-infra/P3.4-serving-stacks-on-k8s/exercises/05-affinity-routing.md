# Exercise 5 — Session-affinity routing in #6 (T0)

Add session affinity to `platform/gateway/router.py`. If a request carries an `X-Session-Id` header, rank backends with **rendezvous (highest-random-weight) hashing** of `(session_id, backend_name)`. The same session then maps to the same backend, and when that backend becomes unhealthy, the session moves to the next-highest one. Only that session moves, so the others keep their caches.

Interface: `Router.candidates(model, session_id=None)`. Wire the header through in `app.py`.

**Test (`test_affinity.py`):** it is skipped until `candidates` accepts `session_id`. It checks that 100 sessions spread across 3 backends (none gets more than 60%), that each session maps to the same backend every time, and that marking one backend unhealthy moves **only** its sessions.
