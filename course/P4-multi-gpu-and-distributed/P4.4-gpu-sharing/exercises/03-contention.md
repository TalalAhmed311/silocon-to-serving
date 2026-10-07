# Exercise 3: the #11 contention matrix (T3)

On a MIG-capable GPU (A100/H100: `p4d.24xlarge`, ideally a Capacity Block. See [aws.md](../aws.md)), run `platform/partitioning/contention.sh` for every mode × tenant count:

| mode | tenants | per-tenant tok/s | p99 ITL ms | Jain |
|---|---|---|---|---|
| exclusive (1 server, 1 GPU) | 1 | | | 1.000 |
| time-slicing | 2 / 4 | | `TODO(run-on: p4d.24xlarge)` | |
| MPS | 2 / 4 | | | |
| MIG (3g + 3g, then 4 × 1g) | 2 / 4 | | | |

**Write up:** which mode gives the best aggregate throughput, which the best p99 ITL, and which the best isolation? Repeat the 4-tenant case with **one** tenant at 4× the request rate of the others. Which mode keeps the other three's p99 ITL stable?
