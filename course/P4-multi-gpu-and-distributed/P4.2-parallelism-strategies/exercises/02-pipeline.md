# Exercise 2: pipeline bubble, formula vs simulation (T0)

`parallel.simulate_pipeline(stages, microbatches, t_fwd, t_bwd, schedule)` is an event simulation with stage dependencies (F needs the previous stage's F, B needs the next stage's B).

1. `test_pipeline.py` checks that the simulated bubble equals `(p−1)/(m+p−1)` for GPipe **and** 1F1B, and that 1F1B holds at most p microbatches in flight while GPipe holds m. Make sure you can explain both results on paper for p = 2, m = 2.
2. **Change the assumption.** Make stage times unequal: stage 0 is 1.5× slower (an embedding-heavy first stage). Add a `t_stage` multiplier list to the simulator. How does the bubble change? Which stage sets the makespan?
3. Write a test for **interleaved 1F1B** (v virtual stages per GPU). The literature says it divides the bubble by v. Confirm it with your simulator, or explain why your version doesn't.
