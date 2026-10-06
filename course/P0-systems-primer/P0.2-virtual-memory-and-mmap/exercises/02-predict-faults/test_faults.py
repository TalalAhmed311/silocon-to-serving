from s2s.exercise import load_impl

m = load_impl(__file__, "faults")
MiB = 1 << 20


def test_sequential_bytes():
    """Touch every 64th byte of 1 MiB: 256 pages, so 256 faults without fault-around."""
    assert m.count_faults(range(0, MiB, 64)) == 256


def test_one_byte_per_page_with_fault_around():
    """One byte per 4 KiB page over 1 MiB with 16-page fault-around: 256 / 16 = 16 faults."""
    assert m.count_faults(range(0, MiB, 4096), around=16) == 16


def test_strided_descending():
    """Touch pages backwards: grouping is by aligned group, so order doesn't change the count."""
    assert m.count_faults(range(MiB - 4096, -1, -4096), around=16) == 16


def test_repeat_is_free():
    assert m.count_faults([0, 1, 2, 4095, 0, 100]) == 1
