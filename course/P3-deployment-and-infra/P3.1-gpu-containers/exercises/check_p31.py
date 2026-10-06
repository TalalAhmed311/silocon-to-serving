"""check_p31.py — verify exercise 2's recorded sizes (T0)."""
import json
import sys

r = json.load(open(sys.argv[1]))
cut = 1 - r["after_bytes"] / r["before_bytes"]
ok = cut >= 0.30 and r.get("changes")
print(f"size reduction: {100 * cut:.0f}% ({'OK' if ok else 'need >= 30% and a list of changes'})")
sys.exit(0 if ok else 1)
