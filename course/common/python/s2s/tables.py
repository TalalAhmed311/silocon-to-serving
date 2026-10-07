"""Markdown table formatting shared by every Python bench and calculator."""
from __future__ import annotations


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1e-2 or abs(v) >= 1e5 else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)
