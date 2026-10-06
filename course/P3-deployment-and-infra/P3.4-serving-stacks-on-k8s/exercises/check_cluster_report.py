"""check_cluster_report.py — the #1 report has every required section and no placeholder left (T0)."""
import re
import sys

text = open(sys.argv[1]).read()
need = ["## Setup", "## Architecture", "## Health checks", "## Results", "## Rollout", "## Teardown"]
problems = [f"missing section {s}" for s in need if s not in text]
if "```mermaid" not in text:
    problems.append("Architecture needs a ```mermaid diagram")
if re.search(r"\bTODO\b|____", text):
    problems.append("report still contains TODO/____ placeholders")
if not re.search(r"[0-9a-f]{7,40}", text):
    problems.append("no commit SHA found: pin versions in Setup")
print("\n".join(problems) or "report complete")
sys.exit(1 if problems else 0)
