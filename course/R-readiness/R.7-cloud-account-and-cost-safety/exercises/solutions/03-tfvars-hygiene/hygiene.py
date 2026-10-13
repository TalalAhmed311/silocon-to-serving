"""Exercise 3 solution."""
from __future__ import annotations

import fnmatch
import re

KEY_ID = re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")
SECRET_ASSIGN = re.compile(r"^\s*([A-Za-z0-9_]*(password|secret|token)[A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", re.I)
SECRET_KEY_NAME = re.compile(r"^\s*aws_secret_access_key\s*=", re.I)


def _is_literal(value: str) -> bool:
    value = value.strip()
    if value in ('""', "''"):
        return False
    return value.startswith(('"', "'")) and len(value) > 2


def check_tfvars(text: str) -> list[str]:
    findings = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if KEY_ID.search(line):
            findings.append(f"line {n}: looks like an AWS access key id")
        if SECRET_KEY_NAME.match(line):
            findings.append(f"line {n}: aws_secret_access_key must never be in a file")
            continue
        m = SECRET_ASSIGN.match(line)
        if m and _is_literal(m.group(3)):
            findings.append(f"line {n}: literal value for {m.group(1)}")
    return findings


def _matches(pattern: str, path: str) -> bool:
    directory_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    anchored = pattern.startswith("/") or "/" in pattern
    pattern = pattern.lstrip("/")
    parts = path.split("/")
    if anchored:
        candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
        if directory_only:
            candidates = candidates[:-1]
        return any(fnmatch.fnmatchcase(c, pattern) for c in candidates)
    names = parts[:-1] if directory_only else parts
    return any(fnmatch.fnmatchcase(p, pattern) for p in names)


def gitignore_covers(lines: list[str], path: str) -> bool:
    ignored = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        if _matches(line, path):
            ignored = not negate
    return ignored
