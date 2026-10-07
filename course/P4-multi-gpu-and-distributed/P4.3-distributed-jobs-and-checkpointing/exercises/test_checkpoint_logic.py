"""Exercise 1 (part a): commit markers, latest discovery, pruning — pure Python, no torch needed."""
from training import checkpoint as ck


def mk(root, step, committed=True):
    d = ck.step_dir(root, step)
    d.mkdir(parents=True)
    (d / "__0_0.distcp").write_bytes(b"x")
    if committed:
        ck.commit(root, step)


def test_latest_ignores_uncommitted(tmp_path):
    mk(tmp_path, 19); mk(tmp_path, 39); mk(tmp_path, 59, committed=False)    # noqa: E702 — crashed mid-save at 59
    assert ck.committed_steps(tmp_path) == [19, 39]
    assert ck.latest_committed(tmp_path) == 39
    assert ck.latest_committed(tmp_path / "missing") is None


def test_prune_keeps_newest_and_drops_stale_partials(tmp_path):
    for s in (19, 39, 59, 79):
        mk(tmp_path, s)
    mk(tmp_path, 69, committed=False)          # older than the newest commit → garbage
    mk(tmp_path, 99, committed=False)          # newer than the newest commit → may still be in progress: keep
    gone = ck.prune(tmp_path, keep=2)
    assert sorted(gone) == [19, 39, 69]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000059", "step_000079", "step_000099"]
