import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
from bakeoff.table import build  # noqa: E402


def fake(dirpath: Path, name: str, vram: int, tpot: float, score: float):
    ev = dirpath / f"{name}_eval" / "m"
    ev.mkdir(parents=True)
    (ev / "results_2026.json").write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}))
    load = {"rows": [{"ttft_p50": 0.2, "tpot_p50": tpot}], "knee": {"offered_rps": 4.0}}
    (dirpath / f"{name}.json").write_text(json.dumps({"variant": name, "vram_used_mib": vram, "kv_tokens": 1000,
                                                      "load": load, "eval_dir": str(dirpath / f"{name}_eval")}))


def test_table_columns_and_delta(tmp_path):
    fake(tmp_path, "bf16", 20000, 0.060, 0.80)
    fake(tmp_path, "fp8-dynamic", 12000, 0.035, 0.79)
    hdr, rows = build(tmp_path)
    by = {r[0]: dict(zip(hdr, r)) for r in rows}
    assert by["fp8-dynamic"]["VRAM MiB"] == 12000
    assert abs(by["fp8-dynamic"]["Δgsm8k vs bf16"] - (-0.01)) < 1e-9
    assert abs(by["bf16"]["decode tok/s/seq (lowest rate)"] - 1 / 0.060) < 1e-9
