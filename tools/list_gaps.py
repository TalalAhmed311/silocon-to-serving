#!/usr/bin/env python3
"""List every TODO(run…) and UNVERIFIED marker by module, as Markdown tables for GAPS.md.

Run: python tools/list_gaps.py > /tmp/gaps.md     (then paste sections C and D into GAPS.md)
Hardware: T0.
"""
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"site", ".git", ".venv", "build", "results", "tools"}
EXT = {".md", ".py", ".sh", ".cu", ".cuh", ".yaml", ".yml", ".tf", ".toml", "Makefile"}
RUN = re.compile(r"TODO\(run(?:-on: ([^)`]*))?\)")


def unit(p: Path) -> str:
    parts = p.relative_to(ROOT).parts
    if parts[0] in ("course", "laneB-cuda") and len(parts) > 2:
        return "/".join(parts[:3]) if parts[1] != "capstone" else "/".join(parts[:3])
    return "/".join(parts[:2]) if len(parts) > 2 else "/".join(parts[:1])


def main() -> None:
    runs, unverified = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or SKIP & set(f.relative_to(ROOT).parts) or (f.suffix not in EXT and f.name not in EXT):
            continue
        if f.name in ("GAPS.md", "course-builder-agent-prompt.md"):
            continue
        text = f.read_text(errors="ignore")
        for m in RUN.finditer(text):
            hw = (m.group(1) or "T0 (not yet executed)").strip()
            if "{" in hw or hw in ("…", "<hardware>"):
                continue
            runs[unit(f)][hw] += 1
        unverified[unit(f)] += text.count("UNVERIFIED")
    print("| Where | Runs needed (hardware × count) |\n|---|---|")
    for u in sorted(runs):
        print(f"| `{u}` | " + ", ".join(f"{hw} × {n}" for hw, n in sorted(runs[u].items())) + " |")
    print("\n| Where | UNVERIFIED markers |\n|---|---|")
    for u in sorted(k for k, v in unverified.items() if v):
        print(f"| `{u}` | {unverified[u]} |")


if __name__ == "__main__":
    main()
