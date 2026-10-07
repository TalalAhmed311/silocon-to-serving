"""run.py — #7 bakeoff: for each variant in variants.yaml: start vLLM, record VRAM + KV capacity from the log, run
#4 (loadgen) at fixed rates, run lm-eval for quality, stop the server. Writes results/bakeoff/<variant>.json.
T2. Run on the GPU instance from the repo root:
    python platform/bakeoff/run.py platform/bakeoff/variants.yaml [--only bf16,fp8-dynamic]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform"))


def serve(model: str, flags: list[str], max_len: int, log: Path) -> subprocess.Popen:
    cmd = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", "8000", "--api-key", os.environ["S2S_API_KEY"],
           "--max-model-len", str(max_len), *flags]
    p = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT)
    H = {"Authorization": f"Bearer {os.environ['S2S_API_KEY']}"}
    for _ in range(900):
        if p.poll() is not None:
            raise RuntimeError(f"vLLM exited; see {log}")
        try:
            if httpx.get("http://127.0.0.1:8000/v1/models", headers=H, timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("vLLM did not start")


def gpu_mem_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    outdir = ROOT / "results/bakeoff"
    outdir.mkdir(parents=True, exist_ok=True)
    for v in cfg["variants"]:
        if a.only and v["name"] not in a.only.split(","):
            continue
        log = outdir / f"{v['name']}.log"
        proc = serve(v["model"], v["serve_flags"], cfg["max_model_len"], log)
        try:
            text = log.read_text(errors="replace")
            m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)       # UNVERIFIED wording (see P1.5)
            rec = {"variant": v["name"], "model": v["model"], "flags": v["serve_flags"], "gpu": cfg["gpu"],
                   "vram_used_mib": gpu_mem_used_mib(), "kv_tokens": int(m.group(1).replace(",", "")) if m else None}
            lg = outdir / f"{v['name']}_load.json"
            subprocess.run([sys.executable, "-m", "loadgen.cli", "--url", "http://127.0.0.1:8000",
                            "--rates", *map(str, cfg["load"]["rates"]), "--duration", str(cfg["load"]["duration"]),
                            "--out", str(lg)], check=True, env={**os.environ, "PYTHONPATH": str(ROOT / "platform")})
            rec["load"] = json.loads(lg.read_text())
            ev = outdir / f"{v['name']}_eval"
            # lm-eval against the running OpenAI-compatible server (local-completions backend); flags UNVERIFIED at 0.4.13
            subprocess.run(["lm_eval", "--model", "local-completions", "--tasks", ",".join(cfg["eval"]["tasks"]),
                            "--limit", str(cfg["eval"]["limit"]), "--output_path", str(ev),
                            "--model_args", f"model={v['model']},base_url=http://127.0.0.1:8000/v1/completions,"
                                            f"num_concurrent=8,tokenized_requests=False"], check=True)
            rec["eval_dir"] = str(ev)
            (outdir / f"{v['name']}.json").write_text(json.dumps(rec, indent=2))
            print("done", v["name"])
        finally:
            proc.terminate()
            proc.wait(timeout=120)


if __name__ == "__main__":
    main()
