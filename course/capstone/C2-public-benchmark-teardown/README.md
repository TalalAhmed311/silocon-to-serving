# C2: Public benchmark teardown (#15)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the report builder and every T0 number. ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange)/![T3](https://img.shields.io/badge/tier-T3%20multi--GPU%20%2F%20two%20regions-red) for the numbers you collected in P2–P6 and C1 |
| **Time** | ≈20 min reading + ≈22 h hands-on (mostly re-running and writing) |
| **Prerequisites** | everything: #0 v1 (P6.7), #2 bake-off and #4 loadgen (P2), #7 quantization, #12 kernels (P5), #13 cost (P3), #14 failover (C1) |
| **You will build** | **#15**: a reproducible public report generated from raw results (`platform/report`), published from your fork |

## What the report is

A stranger with your repo, the stated hardware and a credit card should be able to reproduce every number in it. That rules out three habits: numbers typed by hand, numbers without the configuration that produced them, and leaving out what didn't work.

`platform/report/build.py` assembles `reports/teardown.md` from the JSON your benchmarks wrote, following `platform/report/manifest.yaml`:

- every number is a link to the raw JSON it came from;
- a section without results renders `TODO(run-on: …)` — the builder can't invent a number;
- an appendix lists every input file with its sha256;
- the commit is stamped in the header.

## Structure (fill in around the generated tables)

1. **Summary** — three sentences and the one table people will screenshot (#0 v1 vs vLLM, cost per million tokens).
2. **Architecture** — the platform diagram (gateway #6 → engines → observability, autoscaling, tenancy) in Mermaid, and the versions table: every component's version or SHA from [SOURCES.md](../../../SOURCES.md).
3. **Method** — hardware (instance type, GPU, driver, CUDA), model and dtype, engine flags, workload (`loadgen` args), SLOs, how many runs, and how variance was measured.
4. **Results** — generated sections: serving knee, bake-off, quantization, cost, kernels, failover.
5. **Reproduce** — per section, the exact commands (`make` targets or scripts) and the expected run time and cost. T0 numbers must reproduce **exactly**; T2/T3 within the variance you state.
6. **What didn't work** — dead ends, regressions, numbers that didn't reproduce, and what you'd do next. This is the most-read section of any good teardown.

## Walkthrough

```bash
uv run pytest platform/report -q
PYTHONPATH=platform uv run python -m report.build --root . --out reports/teardown.md
```

Copy each benchmark's JSON into the paths the manifest expects (`results/p6.7/`, `results/bakeoff/`, `results/quant/`, `results/cost/`, `results/kernels/`, `results/c1/`), or edit the manifest to point at where yours are. `results/` is git-ignored, so commit exactly the files the report cites: `git add -f results/<path>.json`. Write `results/c1/gameday.json` with `python -m failover.gameday … ` output saved as JSON (exercise 2).

## Acceptance

- [ ] A clean clone + the README reproduces the T0 numbers exactly (`uv run pytest`, the calculators, the simulators).
- [ ] Every T2/T3 number in the report links to a raw JSON committed to the repo, and its section states hardware and variance.
- [ ] No number appears in the prose that isn't in a generated table (search your text for digits).
- [ ] Prices quote their source and date (EC2 pricing page, region, on-demand or spot).
- [ ] "What didn't work" has at least three entries.

## Exercises

| # | Exercise | Tier |
|---|---|---|
| 1 | Add a section kind to the builder (e.g. `kind: compare` that computes "% of baseline" from two result files, linking both) and its test | T0 |
| 2 | Make `failover.gameday` write JSON (`--json out.json`) and feed it to the report | T0 |
| 3 | Run the reproduction yourself from a fresh clone on fresh instances; record every place the instructions were wrong, and fix them | T2/T3 |
| 4 | *(hard)* Variance: rerun the #4 knee three times on three different instances of the same type; report mean, spread, and whether your engine-vs-vLLM conclusion survives the spread | T2 |

## Common mistakes

- Comparing your best run with someone else's typical run. Same hardware, same limits, same number of runs.
- Cost per token from the knee throughput **at 100 % utilisation**: real fleets run below the knee; state the utilisation you assume (P3.9).
- Quoting a spec-sheet number (TFLOPS, GB/s) as if measured. Cite it as a spec with its source, next to your measured fraction of it.

## Go deeper

- How good systems teardowns are written: vLLM and SGLang release benchmark posts (read their method sections, then look for what they don't say), and MLPerf Inference rules on reporting.
- *The Silicon to Scale* ch. 18 structure for the write-up and the interview story.

**Back to:** [course README](../../../README.md).
