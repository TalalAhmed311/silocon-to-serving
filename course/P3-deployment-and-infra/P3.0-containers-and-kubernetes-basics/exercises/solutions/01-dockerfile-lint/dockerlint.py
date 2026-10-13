"""Exercise 1 solution."""
from __future__ import annotations

import re

INSTALL = re.compile(r"\b(pip3? install|uv sync|uv pip install|npm install|npm ci|poetry install|apt-get install)\b")


def _instructions(text: str) -> list[tuple[int, str, str]]:
    """(first line number, KEYWORD, arguments) with backslash continuations joined."""
    out, buf, start = [], "", 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if not buf:
            start = n
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.partition(" ")
        out.append((start, word.upper(), rest.strip()))
        buf = ""
    if buf:
        word, _, rest = buf.partition(" ")
        out.append((start, word.upper(), rest.strip()))
    return out


def lint(dockerfile_text: str) -> list[str]:
    findings: list[str] = []
    stage_names: set[str] = set()
    copied_all_at = None            # line of `COPY . ...` in the current stage
    last_from_line, last_stage_has_user = 0, False
    for n, kw, args in _instructions(dockerfile_text):
        if kw == "FROM":
            parts = args.split()
            image = next(p for p in parts if not p.startswith("--"))
            if len(parts) >= 3 and parts[-2].upper() == "AS":
                alias = parts[-1].lower()
            else:
                alias = None
            if image.lower() != "scratch" and image.lower() not in stage_names and "@sha256:" not in image:
                last_segment = image.rsplit("/", 1)[-1]
                tag = last_segment.split(":", 1)[1] if ":" in last_segment else ""
                if tag == "":
                    findings.append(f"D1 line {n}: base image has no tag or digest: {image}")
                elif tag == "latest":
                    findings.append(f"D2 line {n}: base image uses the latest tag: {image}")
            if alias:
                stage_names.add(alias)
            copied_all_at = None
            last_from_line, last_stage_has_user = n, False
        elif kw == "COPY" and "--from" not in args:
            src = [p for p in args.split() if not p.startswith("--")]
            if src and src[0] == ".":
                copied_all_at = copied_all_at or n
        elif kw == "RUN" and copied_all_at and INSTALL.search(args):
            findings.append(f"D3 line {copied_all_at}: COPY . before the dependency install on line {n}; copy only the dependency files first")
            copied_all_at = None
        elif kw == "USER":
            user = args.split(":")[0].strip().lower()
            last_stage_has_user = user not in ("root", "0")
    if last_from_line and not last_stage_has_user:
        findings.append(f"D4 line {last_from_line}: the last stage runs as root; add a non-root USER")
    return findings
