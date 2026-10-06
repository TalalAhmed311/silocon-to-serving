# infra/aws/

The Terraform and scripts here are built in Stage 4, at the first T2 module (P2.1). Planned layout and the guardrails every environment gets:

```
single-node/   one GPU EC2 instance: DLAMI, SSM-only (no port 22), gp3/NVMe for weights, IAM read on one S3 prefix
eks/           VPC + terraform-aws-eks v21.26.0, GPU node group or Karpenter, GPU Operator, DCGM, Prometheus/Grafana, KEDA
distributed/   ParallelCluster/Slurm or EKS + EFA for p4d/p5 (following awsome-distributed-training @ d4325212)
guardrails/    AWS Budget alarm, idle auto-stop (CloudWatch alarm on GPU util), tagging
Makefile       up / down / ssm / port-forward / cost
```

Every environment has:

- **Cost:** a stated $/h, verified in-region at build time. Learners re-check it for their own region.
- **A tested teardown:** `make down` (`terraform destroy`).
- **An auto-stop:** stops idle GPU instances after N minutes.
- **Spot:** used only where an interruption is safe.
- **Secrets:** stored in Secrets Manager or SSM Parameter Store; the HF token is passed at runtime.
- **Least-privilege IAM.**
- **No public inference endpoint without auth.**

**Quotas.** G and P instance vCPU quotas often start at 0. Request them through Service Quotas before your first lab; see the guide in P2.1 `aws.md`.
