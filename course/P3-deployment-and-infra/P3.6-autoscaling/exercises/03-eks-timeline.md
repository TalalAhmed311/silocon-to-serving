# Exercise 3: #3 on EKS, load ramp and scale-up timeline (T3)

Follow [aws.md](../aws.md). Deliverables in `results/`:

1. `p3.6-watch.log` → `check_scale_log.py --deployment vllm` prints `OK`.
2. `p3.6-coldstart.md` → the cold-start phase table from `autoscaler.coldstart` for the first pod that needed a **new GPU node**.
3. `p3.6-ramp.json` → the #4 summary. Overlay p90 TTFT over time with the replica timeline (`loadgen.plot` plus the table above). Mark where the cold start shows as a TTFT spike.
4. Apply **one** mitigation from the README §3 table (for example a pre-pull DaemonSet for the vLLM image, or `min_size = 1` on the GPU node group) and repeat 2. Fill both columns of the README bench table.

`TODO(run-on: EKS g6.xlarge)`: expected shape only. Node provisioning and image pull should dominate the first cold start, and weight load the second once the image is cached. Your numbers decide it.
