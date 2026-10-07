"""02_dockerfile_policy.py — a small, opinionated Dockerfile policy check (T0). Used by exercise 1 and CI.

Rules (each is one line in the lesson's §2 table):
  P1 every FROM is pinned: an explicit version tag (not latest/no tag) — and with --strict, a @sha256 digest
  P2 the final stage sets a non-root USER
  P3 no ADD of remote URLs (unverified downloads at build time)
  P4 no ENV/ARG that looks like a secret (TOKEN, SECRET, PASSWORD, API_KEY with a literal value)
  P5 no model weights copied in (COPY of *.safetensors / *.bin / *.gguf)
Run: uv run python course/P3-deployment-and-infra/P3.1-gpu-containers/examples/02_dockerfile_policy.py [--strict] env/Dockerfile.*
Exit code 1 if any violation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SECRET = re.compile(r"^(ENV|ARG)\s+\S*(TOKEN|SECRET|PASSWORD|API_KEY)\S*\s*[= ]\s*\S+", re.I)


def check(text: str, strict: bool = False) -> list[str]:
    problems, stages, user_in_last = [], 0, False
    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        word = line.split()[0].upper()
        if word == "FROM":
            stages += 1
            user_in_last = False
            image = line.split()[1]
            if image.lower() == "scratch" or re.match(r"^[a-z0-9_-]+$", image) and image in {s.lower() for s in ("build",)}:
                continue
            if "@sha256:" not in image:
                tag = image.rsplit(":", 1)[1] if ":" in image.split("/")[-1] else ""
                if tag in ("", "latest"):
                    problems.append(f"line {i}: P1 base image not pinned: {image}")
                elif strict:
                    problems.append(f"line {i}: P1(strict) base pinned by tag only, add @sha256 digest: {image}")
        elif word == "USER":
            user_in_last = line.split()[1] not in ("root", "0")
        elif word == "ADD" and re.search(r"https?://", line):
            problems.append(f"line {i}: P3 ADD from a URL")
        elif SECRET.match(line):
            problems.append(f"line {i}: P4 secret-looking value in {word}")
        elif word == "COPY" and re.search(r"\.(safetensors|bin|gguf|pt)\b", line):
            problems.append(f"line {i}: P5 model weights copied into the image")
    if stages and not user_in_last:
        problems.append("P2 final stage runs as root (add USER <non-root>)")
    return problems


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    files = [a for a in sys.argv[1:] if a != "--strict"]
    bad = 0
    for f in files:
        ps = check(Path(f).read_text(), strict)
        print(f"{f}: {'OK' if not ps else ''}")
        for p in ps:
            print("   ", p)
        bad += bool(ps)
    sys.exit(1 if bad else 0)
