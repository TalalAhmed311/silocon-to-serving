# Exercise 1 — Same model, two stacks, one table (T3)

Deploy the same model, image and flags as (a) the plain Deployment (`overlays/eks`) and (b) KServe RawDeployment (`platform/deploy/kserve/inferenceservice.yaml`; fix field names against KServe v0.21.0's docs if `kubectl apply` rejects them). Run `loadgen.cli` with the same seed and rates against each, and assemble:

| stack | knee req/s | p50 TTFT @ 2 req/s | p90 TPOT @ 2 req/s | pods/objects you had to manage |
|---|---|---|---|---|

Commit the JSONs and the table as `results/p34_compare.md`. Then add Ray Serve if you have time, and explain any gap larger than the run-to-run noise. Measure the noise by repeating one stack 3 times.
