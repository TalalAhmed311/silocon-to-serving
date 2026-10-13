# Exercise 2 — A shell pipeline (easy)

Write `pipeline.sh` so that

```bash
bash pipeline.sh course
```

prints one line per top-level folder of `course` that contains the text `TODO(run`, as `folder count`, largest count first, ties broken by folder name. The count is the number of times the text occurs in all files below that folder. The course marks places that need a real machine with `TODO(run-on: T2)` and `TODO(run)`, so the script tells you where the unrun work is.

Example output (the shape only; your numbers will differ):

```
P5-cuda-deep-dive 14
P1-inference-fundamentals 9
laneB-cuda 3
```

Rules: use pipes (`grep`, `wc`, `sort`). The folder list can come from a shell loop over `*/`, which skips hidden folders and plain files. The script reads its root from `$1` and defaults to `.`.

Try the pieces one at a time in a terminal before you put them in the script:

```bash
grep -rFo 'TODO(run' course/P5-cuda-deep-dive | wc -l       # one folder
sort -k2,2nr -k1,1                                             # sort by column 2 (numeric, reversed), then column 1
```

**Test:** `uv run pytest course/R-readiness/R.3-terminal-git-build/exercises/02-shell-pipeline` builds a small fixture tree in a temporary folder and runs your script on it. Use `S2S_SOLUTIONS=1` to run the reference script.
