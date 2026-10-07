import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("report", ROOT / "course/P2-serving-engines/P2.4-benchmarking-methodology/examples/report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

ROW = {"offered_rps": 2.0, "ok": 120, "goodput_rps": 1.9, "out_tok_s": 250.0, "ttft_p50": 0.2, "ttft_p90": 0.4, "tpot_p90": 0.03}
DATA = {"args": {"seed": 0, "duration": 60, "prompt_mean": 512, "output_mean": 128, "ttft_slo": 2.0, "tpot_slo": 0.1},
        "rows": [ROW], "knee": ROW}
META = {"hardware": "1x NVIDIA L4 GPU (g6.xlarge)", "versions": "vllm 0.31.0", "model": "m@sha bf16", "flags": "--x"}


def test_required_fields_enforced():
    with pytest.raises(ValueError, match="hardware"):
        rep.render(DATA, {**META, "hardware": ""})
    assert "Knee:** 2.00" in rep.render(DATA, META)


def test_small_n_warning():
    pytest.skip("exercise 2.2: delete this line once render() warns on n < 100")
    small = {**DATA, "rows": [{**ROW, "ok": 50}]}
    assert "unreliable" in rep.render(small, META)


def test_hardware_must_name_a_device():
    pytest.skip("exercise 2.3: delete this line once render() validates --hardware")
    with pytest.raises(ValueError):
        rep.render(DATA, {**META, "hardware": "my machine"})
