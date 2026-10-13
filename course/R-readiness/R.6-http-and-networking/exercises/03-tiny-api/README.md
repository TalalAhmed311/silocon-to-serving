# Exercise 3: a tiny JSON API

Fill in the `POST /v1/echo` handler in `app.py`. It reads a JSON body, validates it with pydantic and answers with the right status code: 400 for a body that is not JSON (or blank text), 422 for a body with the wrong shape, 200 with the echoed text otherwise. The test uses FastAPI's `TestClient`, which calls the app in-process with no network, the same way the course's gateway tests do.

```bash
uv run pytest course/R-readiness/R.6-http-and-networking/exercises/03-tiny-api
```
