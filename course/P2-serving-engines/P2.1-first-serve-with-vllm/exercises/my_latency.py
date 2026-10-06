"""Exercise 4 starter: measure end-to-end latency of N identical requests over HTTP."""
import httpx


def measure_e2e(url: str, model: str, prompt_tokens: int, output_tokens: int, n: int = 10, api_key: str | None = None) -> float:
    """Return the median end-to-end latency in seconds (after one warm-up request)."""
    raise NotImplementedError
