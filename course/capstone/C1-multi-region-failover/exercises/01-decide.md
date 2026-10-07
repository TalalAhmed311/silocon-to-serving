# Exercise 1: failover decision logic (T0)

Read `platform/failover/policy.py`, then write down the decision table **before** you look at the tests: for each combination of (active region, primary up?, secondary up?, primary stable long enough?, inside dwell window?), what should happen and why. Compare with `test_decide_table`.

Then extend `decide` with one rule you think is missing and add its test. Candidates: a manual override ("pin to secondary until I say so"); refusing to fail over if the secondary's **capacity** (ready GPU replicas, an extra input) is below what the primary was serving; a maximum number of switches per hour.

**Then:** `test_controller_no_ping_pong` bounds switches under an oscillating primary. What's the worst case number of switches per hour for your parameters?
