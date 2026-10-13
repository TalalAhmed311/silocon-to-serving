import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = (HERE.parent / "solutions" / HERE.name if os.environ.get("S2S_SOLUTIONS") == "1" else HERE) / "pipeline.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")


def make_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def run(root: Path):
    return subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=30)


def lines(root: Path) -> list[str]:
    r = run(root)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_counts_and_sort_order(tmp_path):
    make_tree(tmp_path, {
        "P1/a/README.md": "TODO(run-on: T2) one\n",
        "P1/b/notes.md": "TODO(run) two\nnothing here\nTODO(run: x) three\n",
        "P5/x.cu": "// TODO(run-on: T2)\n",
        "P0/z.txt": "TODO(run) a TODO(run) b\n",   # two markers on ONE line count twice
    })
    assert lines(tmp_path) == ["P1 3", "P0 2", "P5 1"]


def test_ties_break_by_folder_name(tmp_path):
    make_tree(tmp_path, {"b/f": "TODO(run)\n", "a/f": "TODO(run)\n", "c/f": "TODO(run)\nTODO(run)\n"})
    assert lines(tmp_path) == ["c 2", "a 1", "b 1"]


def test_folders_without_markers_are_left_out(tmp_path):
    make_tree(tmp_path, {"has/f": "TODO(run)\n", "none/f": "nothing\n", "empty/.keep": ""})
    assert lines(tmp_path) == ["has 1"]


def test_ignores_hidden_folders_and_root_files(tmp_path):
    make_tree(tmp_path, {".git/x": "TODO(run)\n", "top.md": "TODO(run)\n", "d/f": "TODO(run)\n"})
    assert lines(tmp_path) == ["d 1"]


def test_empty_tree_prints_nothing(tmp_path):
    assert lines(tmp_path) == []


def test_does_not_match_other_todos(tmp_path):
    make_tree(tmp_path, {"d/f": "TODO: later\nTODO(fix)\nTODO (run)\n"})
    assert lines(tmp_path) == []
