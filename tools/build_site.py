#!/usr/bin/env python3
"""Assemble site/docs from the repo's Markdown, then build or serve with MkDocs.

Run: python tools/build_site.py            # build into site/build (strict)
     python tools/build_site.py serve      # live preview
Why a copy step: MkDocs refuses a docs_dir that is the parent of mkdocs.yml, and the
course lives at the repo root so it reads well on GitHub too.
"""
import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "site" / "docs"
INCLUDE = ["README.md", "SOURCES.md", "GLOSSARY.md", "GAPS.md", "PROGRESS.md",
           "course", "laneB-cuda", "platform", "animations", "reports", "infra"]

def assemble() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            # Only Markdown and HTML (animations); code stays in the repo, linked from lessons.
            for f in src.rglob("*"):
                if f.suffix in {".md", ".html", ".png", ".svg", ".json"} and f.is_file():
                    dst = DOCS / f.relative_to(ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
        elif src.exists():
            DOCS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, DOCS / name)

if __name__ == "__main__":
    assemble()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    args = ["mkdocs", cmd, "-f", str(ROOT / "site" / "mkdocs.yml")]
    if cmd == "build":
        args.append("--strict")
    sys.exit(subprocess.call(args))
