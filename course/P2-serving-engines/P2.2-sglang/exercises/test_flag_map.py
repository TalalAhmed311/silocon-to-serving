from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
VLLM, SGL = HERE / "help/vllm-0.31.0.txt", HERE / "help/sglang-0.5.21.txt"


@pytest.mark.skipif(not (VLLM.exists() and SGL.exists()), reason="capture the --help text first (see 01-flag-map.md)")
def test_every_mapped_flag_exists():
    v, s = VLLM.read_text(), SGL.read_text()
    missing = []
    for concept, (vf, sf) in yaml.safe_load((HERE / "flag_map.yaml").read_text()).items():
        if vf != "--model" and vf not in v:
            missing.append(f"vllm {vf} ({concept})")
        if sf not in s:
            missing.append(f"sglang {sf} ({concept})")
    assert not missing, "update flag_map.yaml: " + ", ".join(missing)
