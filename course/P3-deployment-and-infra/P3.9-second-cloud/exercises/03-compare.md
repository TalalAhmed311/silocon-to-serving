# Exercise 3: cross-cloud latency and cost table (T0 with your numbers / T3)

1. Write `results/clouds.yaml` (format in `compare_clouds.py`'s docstring) with **your** prices for `g6.xlarge` and `g2-standard-4` in your regions. Include the date and source page: the script refuses undated prices. Point `loadgen:` at your #4 results from each cloud, or give `tok_s` directly.
2. `PYTHONPATH=platform uv run python course/P3-deployment-and-infra/P3.9-second-cloud/exercises/compare_clouds.py results/clouds.yaml` and paste the table into the README's bench section.
3. **Write up** (5–10 sentences): which cloud wins at 100% utilization, and does that still hold at 30%? Did the knees match (same GPU)? If not, why? How does the cold start difference (P3.6) change the autoscaling policy you'd use on each? What would a 15% committed-use or savings-plan discount do to the ranking?

**Extension:** implement `GCSStore` for `platform/weights/cache.py` (same two methods as `S3Store`, using `google-cloud-storage` with Workload Identity) and add the GCS row to the P3.7 bench.

`TODO(run-on: EKS + GKE)`
