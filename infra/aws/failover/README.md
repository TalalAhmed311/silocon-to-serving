# infra/aws/failover — Route 53 active-passive failover (C1, #14)

Two regions, each running the platform from [`../eks`](../eks/README.md) (gateway + engines + the in-region prober), and one DNS name that points at whichever region is healthy.

```
client ──DNS──▶ llm.example.com ──failover record──▶ primary LB  (us-east-1)  ◀── prober → CloudWatch RegionUp → alarm ─┐
                                   (PRIMARY/SECONDARY)  secondary LB (us-west-2) ◀── prober → CloudWatch RegionUp → alarm ─┤
                                                                                 Route 53 health checks read the alarms ◀─┘
```

| | |
|---|---|
| Cost | Route 53: hosted zone $/month + per-health-check $/month (CloudWatch-metric checks are priced as basic checks; **look both up on the Route 53 pricing page**: `____`). CloudWatch: custom metrics and alarms ($/metric-month, $/alarm-month; look up: `____`). The two EKS stacks dominate: see [`../eks`](../eks/README.md). Plan the drill as **one 4-hour window**; destroy both EKS stacks afterwards |
| Safety | gateway LBs accept only `allowed_client_cidrs` (your IPs) and every request needs a tenant API key; **no unauthenticated endpoint is public** — health comes from the authenticated in-region prober via CloudWatch. Probe key in Secrets Manager |
| Teardown | `make down` here, then `make down` in each region's `../eks`. Check Route 53 for leftover health checks (they bill monthly) |

## Steps

```bash
# 1. two regions (separate state dirs or workspaces), each: ../eks make up && make addons, deploy platform/deploy/overlays/eks
# 2. run the prober in each region (platform/failover/prober.py; a CronJob or Deployment with IRSA for cloudwatch:PutMetricData)
# 3. here:
cp -n terraform.tfvars.example terraform.tfvars    # zone, record, both LB DNS names
make up && make status                              # both health checks: Success
# 4. game day: start the outside client (C1 exercise 3), then
make drill-kill                                     # writes .kill_t
# watch: alarm → health check → DNS answer changes; the client log shows the gap
make drill-restore
make down
```

`TODO(run-on: AWS, two regions)`. `make check` (fmt + validate) runs in CI with the other stacks (`tools/ci_terraform.sh`).

Why CNAME failover records and not alias records? The gateway LBs are created by Kubernetes, not by this stack; a CNAME needs only their DNS names. With alias records you'd also need each LB's hosted zone id (and could use `evaluate_target_health`) — a fine upgrade once the LBs are Terraform-managed.
