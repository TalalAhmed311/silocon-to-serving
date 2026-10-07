# Exercise 4: #9 on T3, with an RTO/RPO table

Run #9 as a RayJob on EKS ([aws.md](../aws.md)) with 4 GPUs (one `g6.12xlarge` worker), then inject three failures:

| failure | how | expected recovery |
|---|---|---|
| process crash | `--die-at-step` (or `kill -9` of one worker process) | Ray restarts the worker group. Resume from the last commit |
| pod loss | `examples/04_fault_inject.sh ray` | KubeRay recreates the pod. Resume from the last commit. Checkpoints must be on shared storage or in S3 |
| node loss | terminate the EC2 instance from the console | the node group launches a replacement: the slowest RTO |

For each, fill in:

| failure | failed at step | resumed from | RPO (steps / s) | RTO (s) | what dominated the RTO |
|---|---|---|---|---|---|
| process | | | | | `TODO(run-on: EKS g6.12xlarge)` |
| pod | | | | | |
| node | | | | | |

`python -m training.rto run1.log run2.log --restart-s <measured>` computes the first columns. **Write up:** which failure would you protect against with more frequent checkpoints, and which with warm spare capacity?
