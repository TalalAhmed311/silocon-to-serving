# Game-day runbook: region failover (template)

Copy this file to your notes, fill it in **before** the drill, and fill the timeline **during** it.

## Scope
- System: gateway + engines in `<primary>` and `<secondary>`, DNS name `<record>`.
- Scenario: primary gateway scaled to zero (`make drill-kill`). Out of scope: data corruption, credential compromise.

## Targets (write before)
| | target |
|---|---|
| detection | |
| client-side failover (RTO) | |
| error rate during drill | |
| budget overspend | 0 |

## Roles
- Incident commander: · Operator (runs commands): · Scribe (timeline): · Observer (dashboards):

## Pre-checks (all must pass)
- [ ] both health checks `Success` (`make status`)
- [ ] secondary has ≥ N warm engine replicas and weights pre-staged
- [ ] outside client logging; dashboards open (P3.5): request rate per region, error rate, TTFT
- [ ] probe tenant key valid in both regions; budget sync running (if active-active)
- [ ] abort criteria agreed: e.g. secondary error rate > 20 % for 5 min → restore primary

## Procedure
1. T0: `make drill-kill` (records `.kill_t`).
2. Watch: alarm state, health check, `dig +short <record>` every 10 s, client log.
3. When the client shows 20 consecutive successes in the secondary: declare failed over.
4. Hold 10 min. Then `make drill-restore`; observe failback per policy.
5. Stop the client; run `python -m failover.gameday …`.

## Timeline (fill during)
| time (UTC) | event | source |
|---|---|---|
| | kill | `.kill_t` |
| | first client error | client log |
| | alarm ALARM | CloudWatch |
| | health check unhealthy | Route 53 |
| | DNS answer changed | dig |
| | first success in secondary | client log |
| | steady in secondary | gameday.py |
| | restore | |
| | failback | |

## Results and follow-ups
- Measured vs targets:
- What surprised us:
- Action items (owner, date):
