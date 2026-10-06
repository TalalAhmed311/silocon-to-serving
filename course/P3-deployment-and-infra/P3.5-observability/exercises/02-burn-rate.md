# Exercise 2 — Burn-rate alerts: prove they fire and reset (T0)

Add a third test to `slo_test.yaml`: 40 minutes at 20% slow requests, then 60 minutes at 0% slow. Assert that:

1. `TTFTBudgetFastBurn` fires by minute ~40
2. it has **stopped** firing ~5–10 minutes after the recovery, because the 5-minute window drops below the threshold even though the 1-hour window is still elevated

That reset behaviour is the reason for the second, short window. Then compute how much of a 30-day error budget those 40 minutes consumed, at 100 req/min with 20% bad: 40 × 100 × 0.2 = 800 bad requests, out of a budget of 1% × 30 × 1440 × 100 = 43,200.
