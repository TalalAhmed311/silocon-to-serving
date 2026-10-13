"""Exercise 1 starter: a beginner Dockerfile linter. Four rules."""
from __future__ import annotations


def lint(dockerfile_text: str) -> list[str]:
    """Return findings as strings that start with a rule code and a line number.

    D1 line N: a FROM with no tag and no digest ("FROM python")
    D2 line N: a FROM that uses the tag "latest"
    D3 line N: `COPY . .` (or `COPY . /something`) appears BEFORE a dependency install in the
               same stage, so any code change invalidates the install layer
               (dependency installs: pip install, uv sync, uv pip install, npm install, npm ci,
               poetry install, apt-get install)
    D4 line N: the last stage never sets a non-root USER (report the line of its FROM)

    Rules for parsing:
    - skip blank lines and comments
    - join lines that end with a backslash into one instruction
    - `FROM scratch` and `FROM <earlier stage name>` are not D1/D2 findings
    - an image with a @sha256: digest is pinned, whatever its tag
    - a tag other than latest (python:3.12-slim) is accepted
    - USER root and USER 0 do not count as non-root
    """
    raise NotImplementedError
