# Exercise 3: S3 throughput vs concurrency (T2/T3)

On the instance (see [aws.md](../aws.md)), run `weights.bench` with `--s3 … --concurrency 1 2 4 8 16 32 64`. It writes `results/p3.7-loader.json`.

```bash
python course/P3-deployment-and-infra/P3.7-weight-delivery/exercises/check_s3_curve.py results/p3.7-loader.json --instance-gbps <your instance's network Gbps>
```

The checker prints the `concurrency | GB/s` curve and:

- **fails** if throughput at the highest concurrency isn't at least 2× the single-stream rate (you're likely limited by the destination disk: check `iostat`)
- reports the knee: the smallest concurrency within 10% of the best result
- reports the best result as a fraction of the instance network bandwidth you passed in

**Write up:** which hop is the plateau? Network, EBS gp3 throughput, or instance-store NVMe? Prove it with a second run to `/dev/shm`, or to a different volume.

`TODO(run-on: g6.xlarge)`
