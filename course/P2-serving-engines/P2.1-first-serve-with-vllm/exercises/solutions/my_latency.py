"""Exercise 4 solution."""
import statistics
import sys
import time

import httpx


def measure_e2e(url, model, prompt_tokens, output_tokens, n=10, api_key=None):
    H = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "prompt": " ".join(["hello"] * prompt_tokens), "max_tokens": output_tokens,
            "temperature": 0, "ignore_eos": True}       # ignore_eos: always generate exactly output_tokens
    with httpx.Client(timeout=600) as c:
        c.post(f"{url}/v1/completions", json=body, headers=H).raise_for_status()   # warm-up
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = c.post(f"{url}/v1/completions", json=body, headers=H)
            r.raise_for_status()
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    model = httpx.get(f"{url}/v1/models").json()["data"][0]["id"]
    print(f"median e2e: {measure_e2e(url, model, 128, 128):.3f} s")
