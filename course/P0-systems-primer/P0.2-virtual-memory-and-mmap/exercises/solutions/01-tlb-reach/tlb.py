"""Exercise 1 solution."""

MY_CPU_L2_TLB_ENTRIES = 1536  # example value; the test does not depend on it


def tlb_reach(entries: int, page_bytes: int) -> int:
    return entries * page_bytes


def pages_needed(working_set_bytes: int, page_bytes: int) -> int:
    return -(-working_set_bytes // page_bytes)  # ceil division without floats


def fits_in_tlb(working_set_bytes: int, entries: int, page_bytes: int) -> bool:
    return pages_needed(working_set_bytes, page_bytes) <= entries


def walks_saved_by_hugepages(working_set_bytes: int, small: int = 4096, huge: int = 2 << 20) -> int:
    return pages_needed(working_set_bytes, small) - pages_needed(working_set_bytes, huge)


if __name__ == "__main__":
    for page in (4096, 2 << 20):
        print(f"page {page:>8} B: reach = {tlb_reach(MY_CPU_L2_TLB_ENTRIES, page) / 2**20:.0f} MiB")
