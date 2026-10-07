#!/usr/bin/env python3
"""Check Markdown links in the repo.

Run: python tools/check_links.py            # relative links only (offline)
     python tools/check_links.py --external # also HEAD-request http(s) links
Expected output: "OK: N files, M relative links" or a list of broken links and exit code 1.
Hardware: T0.
"""
import re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".git", "site", "node_modules", ".venv"}


def md_files():
    for f in ROOT.rglob("*.md"):
        if not SKIP_DIRS.intersection(f.relative_to(ROOT).parts):
            yield f


def main(external: bool) -> int:
    broken, n_rel, n_files, ext = [], 0, 0, set()
    for f in md_files():
        n_files += 1
        # Fenced code blocks contain example syntax, not real links.
        text = re.sub(r"```.*?```", "", f.read_text(encoding="utf-8"), flags=re.S)
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://")):
                ext.add(target)
                continue
            if target.startswith(("#", "mailto:")):
                continue
            n_rel += 1
            path = (f.parent / target.split("#")[0]).resolve()
            if not path.exists():
                broken.append(f"{f.relative_to(ROOT)} -> {target}")
    if external:
        for url in sorted(ext):
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "s2s-linkcheck"})
                urllib.request.urlopen(req, timeout=15)
            except Exception as e:  # report, don't crash: some hosts reject HEAD
                broken.append(f"{url} ({e})")
    for b in broken:
        print("BROKEN:", b)
    print(f"{'FAIL' if broken else 'OK'}: {n_files} files, {n_rel} relative links, {len(ext)} external links"
          + ("" if external else " (external not checked)"))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main("--external" in sys.argv))
