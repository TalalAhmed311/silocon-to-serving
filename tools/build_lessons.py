#!/usr/bin/env python3
"""build_lessons.py — write lessons/<id>.html for every lessons/content/<id>.js, and assets/available.js.

Each lesson page is a thin wrapper: shared fonts + lesson.css + GSAP + lesson.js, then the lesson's content file.
Run after adding or renaming a content file:   python tools/build_lessons.py
Hardware: T0.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "lessons"
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800"
         "&family=Manrope:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap")
GSAP = "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title}</title>
<link rel="stylesheet" href="{fonts}">
<link rel="stylesheet" href="assets/lesson.css">
<script src="{gsap}"></script>
<script src="assets/lesson.js"></script>
</head>
<body>
<div id="app"></div>
<script src="content/{id}.js"></script>
</body>
</html>
"""


def main() -> None:
    ids = []
    for f in sorted((ROOT / "content").glob("*.js")):
        src = f.read_text()
        n = re.search(r'\bn:\s*"([^"]+)"', src)
        t = re.search(r'\btitle:\s*"([^"]+)"', src)
        title = f"{n.group(1)} {t.group(1)}" if n and t else f.stem
        (ROOT / f"{f.stem}.html").write_text(PAGE.format(title=title, fonts=FONTS, gsap=GSAP, id=f.stem))
        ids.append(f.stem)
    (ROOT / "assets" / "available.js").write_text("window.S2S_AVAILABLE = " + json.dumps(ids) + ";\n")
    print(f"built {len(ids)} lesson pages: {', '.join(ids)}")


if __name__ == "__main__":
    main()
