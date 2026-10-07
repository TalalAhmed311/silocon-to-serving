"""cache_bench.py — ms per generated token with and without a KV cache, as the sequence grows (NumPy, tiny model).

Run:      uv run python course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/bench/cache_bench.py
Output:   | seq len | no-cache ms/token | cache ms/token | speedup | + results/cache.json
Hardware: T0. Median of 5 repeats per point.
"""
import importlib.util
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "course/common/python"))
from s2s.bench import save_results  # noqa: E402
from s2s.tables import md_table  # noqa: E402

V0 = ROOT / "platform/engine/v0"
spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference/llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

d = tempfile.mkdtemp()
subprocess.run([sys.executable, str(V0 / "tools/make_tiny_llama.py"), d, "--max-seq", "512"], check=True, capture_output=True)
rows, js = [], []
for n in (16, 64, 128, 256):
    toks = [i % 200 + 1 for i in range(n)]
    nc, c = [], []
    for _ in range(5):
        m = ref.LlamaNumpy(d)
        for p in range(n - 1):
            m.forward(toks[p], p)
        t0 = time.perf_counter(); m.forward(toks[-1], n - 1); c.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        fresh = ref.LlamaNumpy(d)
        for p in range(n):                     # recompute the whole prefix for the last token
            fresh.forward(toks[p], p)
        nc.append((time.perf_counter() - t0) * 1e3)
    a, b = statistics.median(nc), statistics.median(c)
    rows.append([n, a, b, f"{a / b:.0f}x"])
    js.append({"label": "speedup", "size": n, "rate": a / b, "median_ms": b})
print(md_table(["seq len", "no-cache ms/token", "cache ms/token", "speedup"], rows))
print("wrote", save_results("cache", "x", js))
