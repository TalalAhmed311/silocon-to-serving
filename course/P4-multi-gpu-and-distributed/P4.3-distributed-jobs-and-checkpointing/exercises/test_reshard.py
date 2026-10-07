"""Exercise 2: resume with a different world size (2 → 4 processes, CPU/gloo). Slow: spawns torchrun 3 times.

Uninterrupted world-2 run to step 59 vs (world 2 to step 39, then world 4 from the checkpoint to 59): the global batch
and the data stream are identical by construction, so the final losses must agree to float tolerance."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytestmark = [pytest.mark.torch, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "platform/training/train.py"


def run(world, steps, ckpt, port):
    out = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(world),
                          "--master-port", str(port), str(TRAIN), "--steps", str(steps), "--ckpt-every", "20",
                          "--ckpt", str(ckpt), "--bs", "4"], capture_output=True, text=True, timeout=600,
                         env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-3000:]
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def test_two_to_four(tmp_path):
    ref = run(2, 60, tmp_path / "a", 29621)
    run(2, 40, tmp_path / "b", 29622)
    resumed = run(4, 60, tmp_path / "b", 29623)
    assert resumed[0]["resumed_from"] == 39 and resumed[0]["world"] == 4
    assert resumed[-1]["step"] == ref[-1]["step"] == 59
    assert resumed[-1]["loss"] == pytest.approx(ref[-1]["loss"], rel=2e-3)
