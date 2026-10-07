# C1 quiz

<details><summary><b>1.</b> Which serving state can you lose in a region failure without consequence, and which can't you?</summary>

KV and prefix caches and in-flight requests are disposable (recompute, client retry). Tenant keys, budgets and the audit log are not, and weights must already be in the other region for a fast RTO.
</details>

<details><summary><b>2.</b> Route 53 flipped the record 90 s after the kill, but users saw errors for 4 minutes. Name two reasons.</summary>

Client and resolver DNS caching beyond the TTL; clients holding pooled connections to the old region's LB; and the secondary scaling up for the extra load.
</details>

<details><summary><b>3.</b> Why use a CloudWatch-alarm health check instead of a Route 53 HTTP check against /health?</summary>

The HTTP check needs an endpoint reachable from Route 53's public checkers, and a shallow /health can be green while inference is broken. An in-region, authenticated probe that runs a real completion avoids both.
</details>

<details><summary><b>4.</b> Why must "fail back" be slower than "fail over"?</summary>

A just-recovered region is the most likely to fail again; moving traffic back immediately risks ping-pong, and each move costs errors and cold caches.
</details>

<details><summary><b>5.</b> Two regions each see a tenant's remaining budget of 1,000 tokens at the last sync. With a naive check, how much can the tenant spend before the next sync? With the split gate?</summary>

Up to 2,000 with the naive check (each region spends the whole remainder). At most 1,000 with the split gate (500 per region).
</details>
