"""Exercise 1 starter."""

MY_CPU_L2_TLB_ENTRIES = 0  # TODO: look this up for your CPU


def tlb_reach(entries: int, page_bytes: int) -> int:
    raise NotImplementedError  # TODO


def pages_needed(working_set_bytes: int, page_bytes: int) -> int:
    raise NotImplementedError  # TODO (round up!)


def fits_in_tlb(working_set_bytes: int, entries: int, page_bytes: int) -> bool:
    raise NotImplementedError  # TODO


def walks_saved_by_hugepages(working_set_bytes: int, small: int = 4096, huge: int = 2 << 20) -> int:
    raise NotImplementedError  # TODO
