import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("T2: needs an NVIDIA GPU", allow_module_level=True)

from s2s.exercise import load_impl  # noqa: E402

m = load_impl(__file__, "bw")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import specs  # noqa: E402


def test_bandwidth():
    gbs = m.measure_copy_gbs(1 << 30)
    name = torch.cuda.get_device_name()
    spec = next((g for g in specs.load() if g["name"].split("-")[0] in name), None)
    print(f"\n{name}: {gbs:.0f} GB/s (spec {spec['hbm_gbs'] if spec else '?'})")
    assert gbs > (0.6 * spec["hbm_gbs"] if spec and spec["hbm_gbs"] else 0)
