# Exercise 1: insert, match, split (T0)

Implement `match_prefix` and `insert` in `my_radix.py`. The structure and the rules are in README §1.

`test_match_agrees_with_naive_trie` runs 50 random trees × 40 random inserts/matches over a 4-letter alphabet (lots of shared prefixes and splits) and checks: the matched length equals a brute-force longest common prefix against everything inserted; the returned slots are exactly the slots inserted for those positions; `insert` reports the already-cached length; and the tree's structural invariants hold (`check_tree`). `test_split_on_partial_match` is the minimal split case.

**Then:** what's the worst-case depth of the tree after inserting n sequences, and does it matter for the cost of `lock`?
