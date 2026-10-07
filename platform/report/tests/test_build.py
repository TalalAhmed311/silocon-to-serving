"""#15 report builder (C2). T0."""
import json

import yaml

from report.build import HERE, build


def test_missing_results_are_todo_not_numbers(tmp_path):
    text = build(tmp_path, yaml.safe_load(open(HERE / "manifest.yaml")), tmp_path / "reports" / "t.md", commit="abc")
    assert "Built from commit `abc`" in text
    assert text.count("TODO(run-on:") == len(yaml.safe_load(open(HERE / "manifest.yaml"))["sections"])


def test_numbers_link_to_their_json(tmp_path):
    (tmp_path / "results" / "p6.7").mkdir(parents=True)
    (tmp_path / "results" / "p6.7" / "vllm.json").write_text(json.dumps(
        {"args": {"url": "http://127.0.0.1:8000"}, "rows": [], "knee": {"offered_rps": 4.0, "out_tok_s": 512.0, "ttft_p90": 1.2, "tpot_p90": 0.05}}))
    (tmp_path / "results" / "cost").mkdir(parents=True)
    (tmp_path / "results" / "cost" / "g6.json").write_text(json.dumps(
        {"instance": "g6.xlarge", "price_per_h": 1.0, "tok_s": 1000.0, "usd_per_mtok": 0.278, "price_source": "pricing page 2026-10"}))
    text = build(tmp_path, yaml.safe_load(open(HERE / "manifest.yaml")), tmp_path / "reports" / "t.md", commit="x")
    assert "[4](../results/p6.7/vllm.json)" in text
    assert "[50](../results/p6.7/vllm.json)" in text                    # tpot in ms
    assert "[0.278](../results/cost/g6.json)" in text
    assert "pricing page 2026-10" in text
    assert "results/cost/g6.json" in text.split("## Appendix")[1]
    assert text.count("TODO(run-on:") == 4                               # the four sections still without data
