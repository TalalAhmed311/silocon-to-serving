#!/usr/bin/env python3
"""Re-inline animations/src/{core.css,core.js,controls.html} into every animations/*.html.

Run: python tools/build_animations.py
Each animation keeps its own code; only the regions between these markers are replaced:
  /*CORE-CSS*/ ... /*END-CORE-CSS*/      /*CORE-JS*/ ... /*END-CORE-JS*/      <!--CONTROLS--> ... <!--END-CONTROLS-->
Hardware: T0.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "animations"
SRC = ROOT / "src"
PARTS = {
    ("/*CORE-CSS*/", "/*END-CORE-CSS*/"): (SRC / "core.css").read_text(),
    ("/*CORE-JS*/", "/*END-CORE-JS*/"): (SRC / "core.js").read_text(),
    ("<!--CONTROLS-->", "<!--END-CONTROLS-->"): (SRC / "controls.html").read_text(),
}

for page in sorted(ROOT.glob("*.html")):
    text = page.read_text()
    for (start, end), body in PARTS.items():
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        text = pattern.sub(lambda _m: f"{start}\n{body}\n{end}", text)
    page.write_text(text)
    print("rebuilt", page.name)
