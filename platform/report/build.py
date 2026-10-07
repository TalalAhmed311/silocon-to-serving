"""build.py — render the teardown report (C2, #15) from raw result JSON.

    python -m report.build --root . --out reports/teardown.md [--manifest platform/report/manifest.yaml]

Rules it enforces (the acceptance criteria of #15):
  * every number in a table links to the JSON file it came from;
  * a section with no results says TODO(run-on: …) — nothing is estimated or filled in;
  * an appendix lists every input file with its sha256, so a reader can check they reproduced the same raw data;
  * the git commit of the repo is stamped in the header (pass --commit, or it's read from .git).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return "—" if v is None else str(v)


def link(text: str, rel: str) -> str:
    return f"[{text}]({rel})"


def render_loadgen(files: list[Path], rel) -> list[str]:
    out = ["| result | model | knee req/s | knee out tok/s | p90 TTFT at knee (s) | p90 TPOT at knee (ms) |", "|---|---|---|---|---|---|"]
    for f in files:
        d = json.loads(f.read_text())
        k = d.get("knee")
        r = rel(f)
        if not k:
            out.append(f"| {link(f.stem, r)} | {d.get('args', {}).get('url', '')} | knee not reached | | | |")
            continue
        out.append(f"| {link(f.stem, r)} | {d.get('args', {}).get('url', '')} | {link(fmt(k.get('offered_rps')), r)} | "
                   f"{link(fmt(k.get('out_tok_s')), r)} | {link(fmt(k.get('ttft_p90')), r)} | "
                   f"{link(fmt(1e3 * k['tpot_p90']) if k.get('tpot_p90') is not None else '—', r)} |")
    return out


def render_table(files: list[Path], cols: list[str], rel) -> list[str]:
    out = ["| " + " | ".join(cols) + " | source |", "|" + "---|" * (len(cols) + 1)]
    for f in files:
        d = json.loads(f.read_text())
        rows = d if isinstance(d, list) else d.get("rows", [d])
        for row in rows:
            out.append("| " + " | ".join(link(fmt(row.get(c)), rel(f)) if isinstance(row.get(c), (int, float)) else fmt(row.get(c))
                                         for c in cols) + f" | {link(f.name, rel(f))} |")
    return out


def render_kv(files: list[Path], rel) -> list[str]:
    out = ["| metric | value |", "|---|---|"]
    for f in files:
        for k, v in json.loads(f.read_text()).items():
            out.append(f"| {k} | {link(fmt(v), rel(f)) if isinstance(v, (int, float)) else fmt(v)} |")
    return out


def build(root: Path, manifest: dict, out_path: Path, commit: str | None = None) -> str:
    root, out_path = root.resolve(), out_path.resolve()
    out_dir = out_path.parent

    def rel(p: Path) -> str:
        return Path(*([".."] * len(out_dir.resolve().relative_to(root.resolve()).parts)), p.resolve().relative_to(root.resolve())).as_posix()

    if commit is None:
        try:
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    lines = [f"# {manifest['title']}", "", f"Built from commit `{commit}` by `platform/report/build.py`. "
             "Every number links to the raw JSON it came from; sections without results are marked TODO.", ""]
    used: list[Path] = []
    for s in manifest["sections"]:
        files = sorted({Path(p) for g in s["files"] for p in glob.glob(str(root / g))})
        lines += [f"## {s['title']}", "", f"Hardware: {s.get('hardware', 'stated per row')}.", ""]
        if not files:
            lines += [f"`TODO(run-on: {s.get('hardware', '?')})` — no results matching {', '.join(s['files'])}.", ""]
            continue
        used += files
        if s["kind"] == "loadgen":
            lines += render_loadgen(files, rel)
        elif s["kind"] == "table":
            lines += render_table(files, s["columns"], rel)
        elif s["kind"] == "kv":
            lines += render_kv(files, rel)
        else:
            raise ValueError(f"unknown section kind {s['kind']}")
        lines.append("")
    lines += ["## Appendix: inputs", "", "| file | sha256 |", "|---|---|"]
    lines += [f"| {link(f.relative_to(root).as_posix(), rel(f))} | `{sha256(f)[:16]}…` |" for f in used]
    lines += ["", "## What didn't work", "", "_Write this section by hand: dead ends, numbers that didn't reproduce, and why._", ""]
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default=str(HERE / "manifest.yaml"))
    ap.add_argument("--out", default="reports/teardown.md")
    ap.add_argument("--commit")
    a = ap.parse_args()
    root = Path(a.root)
    print(build(root, yaml.safe_load(open(a.manifest)), root / a.out, a.commit)[:400], "…")
