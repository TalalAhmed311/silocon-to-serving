# Exercise 3 — #1 checklist + report (T3)

Complete the checklist in the lesson (§4) and write `results/p34_cluster.md` with these sections:

- `## Setup`: cluster, node types, versions with SHAs, and model@revision
- `## Architecture`: a Mermaid diagram of what you deployed
- `## Health checks`: the probes, and the measured start-up timeline
- `## Results`: P2.4's `report.py` output for your #4 run
- `## Rollout`: the `drain_test.py` result
- `## Teardown`: the `make down` + `cost.sh` output

Then run `uv run python course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s/exercises/check_cluster_report.py results/p34_cluster.md`.
