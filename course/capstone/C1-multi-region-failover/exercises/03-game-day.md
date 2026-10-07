# Exercise 3: the game day (T3)

1. Fill in the [runbook template](../runbook.md) **before** the drill: targets, roles (even if you're all of them), abort criteria, and the exact commands.
2. Bring up both regions and the failover stack (`infra/aws/failover/README.md`). Pre-stage weights in the secondary.
3. From a machine outside both regions, run a client loop that logs one JSON line per request (`{"t", "end", "ok", "region"}` — take `region` from a response header your gateway adds, or from which LB answered).
4. `make drill-kill`. Watch the CloudWatch alarm, the health check, `dig` answers, and the client log. Note the time the alarm and the health check flipped.
5. `make drill-restore`; observe failback (or do it by hand, per your policy).
6. `python -m failover.gameday client.jsonl --kill-t $(cat infra/aws/failover/.kill_t) --switch-t <health check flip time>` → the bench table.

Deliverable: the filled runbook with the measured timeline, the bench table, and a "what surprised us" section. `TODO(run-on: AWS, two regions)`.
