#!/usr/bin/env bash
# Exercise 2 reference solution.
export LC_ALL=C                      # same sort order on every machine
root="${1:-.}"
for dir in "$root"/*/; do            # the glob */ skips hidden folders and plain files
  name=$(basename "$dir")
  n=$(grep -rFo 'TODO(run' "$dir" 2>/dev/null | wc -l)   # -o prints one line per match
  if [ "$n" -gt 0 ]; then echo "$name $n"; fi
done | sort -k2,2nr -k1,1            # count descending, then folder name
