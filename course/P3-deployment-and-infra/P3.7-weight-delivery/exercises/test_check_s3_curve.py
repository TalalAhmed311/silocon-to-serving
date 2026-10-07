import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("check_s3_curve", Path(__file__).with_name("check_s3_curve.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

ROWS = [{"method": "read-warm", "gb": 16, "seconds": 4},
        {"method": "s3-download c=1", "gb": 16, "seconds": 160},
        {"method": "s3-download c=4", "gb": 16, "seconds": 45},
        {"method": "s3-download c=16", "gb": 16, "seconds": 17},
        {"method": "s3-download c=64", "gb": 16, "seconds": 16}]


def test_curve_and_knee(tmp_path):
    pts = m.curve(ROWS)
    assert [c for c, _ in pts] == [1, 4, 16, 64]
    assert m.analyze(pts)["knee"] == 16
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROWS))
    assert m.main([str(p)]) == 0
    p.write_text(json.dumps([dict(r, seconds=160) if r["method"].startswith("s3") else r for r in ROWS]))
    assert m.main([str(p)]) == 1
