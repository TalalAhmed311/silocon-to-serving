import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("compare_clouds", Path(__file__).with_name("compare_clouds.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

# Made-up round inputs: the test checks arithmetic, not prices.
E = [{"cloud": "A", "node": "n1", "usd_per_hour": 1.0, "price_source": "test", "gpus": 1, "tok_s": 1000 / 3.6, "cold_start_s": 90},
     {"cloud": "B", "node": "n2", "usd_per_hour": 4.0, "price_source": "test", "gpus": 2, "tok_s": 1000 / 3.6}]


def test_rows():
    a, b = m.row(E[0]), m.row(E[1])
    assert a["usd_per_m_100"] == pytest.approx(1.0)          # $1/h ÷ 1M tok/h
    assert b["usd_gpu_h"] == pytest.approx(2.0) and b["usd_per_m_100"] == pytest.approx(2.0)
    assert a["usd_per_m_util"] == pytest.approx(1.0 / 0.3)


def test_requires_sourced_prices():
    with pytest.raises(ValueError):
        m.row({**E[0], "price_source": ""})


def test_table_renders():
    t = m.table(E)
    assert t.count("\n") == 3 and "| A | n1 |" in t and "| — |" in t
