# Exercise 1: block-table invariants (T0)

Implement `can_allocate`, `allocate` (without CoW for now), `free_seq` and `fork` in `my_block_manager.py`.

`test_random_ops_keep_invariants` runs 30 random pools × 200 random operations (admit, append a token, fork, free) and calls `check_invariants` after every one: ref counts equal block-table occurrences, and every block is in exactly one of {in use, free, cached-free}. `test_slot_mapping` and `test_no_free_blocks_raises` cover the basics.

**Then:** the reference releases a sequence's blocks tail first. Construct a case where head-first release loses a prefix-cache hit that tail-first keeps.
