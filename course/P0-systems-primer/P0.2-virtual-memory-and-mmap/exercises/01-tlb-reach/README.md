# Exercise 1 — TLB reach calculator (easy)

Implement these in `tlb.py`:

- `tlb_reach(entries, page_bytes) -> int`: bytes covered by a TLB.
- `pages_needed(working_set_bytes, page_bytes) -> int`: rounds up.
- `fits_in_tlb(working_set_bytes, entries, page_bytes) -> bool`.
- `walks_saved_by_hugepages(working_set_bytes, small=4096, huge=2<<20) -> int`: how many fewer translations a random walk over the whole working set needs with huge pages. That is the difference in page counts.

Then look up your own CPU's L2 TLB entry count (Intel/AMD optimization manual, Agner Fog's microarchitecture guide, or `cpuid` / `x86info -c`). Write it in the `MY_CPU_L2_TLB_ENTRIES` constant, and print the reach for 4 KiB and 2 MiB pages.

**Test:** `uv run pytest exercises/01-tlb-reach`
