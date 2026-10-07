# Exercise 4 (hard): active-active with budget reconciliation

**T0.** `platform/failover/budgets.py` has a G-counter and two admission gates. Using `simulate()` from the tests:

1. Measure overspend for the naive gate as a function of sync interval (10, 50, 200 steps) and number of regions (2, 3).
2. Measure the **refusal rate** of the split gate when traffic is skewed (90 % of requests to one region) — requests refused while the tenant still had budget left globally.
3. Design a better split: allowances proportional to each region's recent share of the tenant's traffic. Prove (in a test) it still never overspends, and show its refusal rate on the skewed workload.

**T3.** Wire it into the gateway: each region publishes its counter state (e.g. to a DynamoDB global table or S3 every N seconds) and merges the others'. Kill one region mid-drill: what happens to its unsynced spend, and is that an under- or over-count?
