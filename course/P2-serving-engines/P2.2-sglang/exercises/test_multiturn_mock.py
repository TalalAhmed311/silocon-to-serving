import asyncio
import importlib.util
from pathlib import Path

import httpx

from s2s.mockserver import running_mock

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("mt", ROOT / "course/P2-serving-engines/P2.2-sglang/examples/02_multiturn.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def test_ttft_grows_without_prefix_cache():
    # prefill cost dominates: 0.5 ms per prompt "token" (whitespace word) in the mock
    with running_mock(PREFILL_MS_PER_TOKEN=0.5, BASE_STEP_MS=1, PER_SEQ_MS=0) as url:
        async def go():
            ttfts = [[] for _ in range(4)]
            async with httpx.AsyncClient() as c:
                await mt.conversation(c, url, {}, "mock", 4, 400, 16, ttfts)
            return [t[0] for t in ttfts]
        t = asyncio.run(go())
    assert t[3] > t[0], f"TTFT should grow with history on a cache-less engine: {t}"
