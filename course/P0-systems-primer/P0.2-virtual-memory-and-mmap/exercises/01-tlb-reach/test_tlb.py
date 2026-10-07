from s2s.exercise import load_impl

m = load_impl(__file__, "tlb")
KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30


def test_reach():
    assert m.tlb_reach(64, 4 * KiB) == 256 * KiB
    assert m.tlb_reach(1536, 2 * MiB) == 3 * GiB


def test_pages_round_up():
    assert m.pages_needed(1, 4 * KiB) == 1
    assert m.pages_needed(4 * KiB, 4 * KiB) == 1
    assert m.pages_needed(4 * KiB + 1, 4 * KiB) == 2
    assert m.pages_needed(16 * GiB, 4 * KiB) == 4 * MiB  # 16 GiB of 7B-fp16 weights = 4 Mi pages


def test_fits():
    assert m.fits_in_tlb(6 * MiB, 1536, 4 * KiB)
    assert not m.fits_in_tlb(6 * MiB + 1, 1536, 4 * KiB)


def test_hugepage_savings():
    # 1 GiB: 262144 small pages vs 512 huge pages
    assert m.walks_saved_by_hugepages(GiB) == 262144 - 512
