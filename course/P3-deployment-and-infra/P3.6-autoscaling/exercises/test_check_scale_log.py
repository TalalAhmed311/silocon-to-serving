"""The checker itself is tested on a synthetic log, so you can trust it on your real one."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("check_scale_log", Path(__file__).with_name("check_scale_log.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

LOG = """\
100.0 ADDED    mockllm-aa-1  1/1  Running  0  5m
130.0 ADDED    mockllm-aa-2  0/1  Pending  0  0s
131.0 MODIFIED mockllm-aa-2  0/1  ContainerCreating  0  1s
140.0 MODIFIED mockllm-aa-2  1/1  Running  0  10s
300.0 MODIFIED mockllm-aa-2  0/1  Terminating  0  3m
305.0 DELETED  mockllm-aa-2  0/1  Terminating  0  3m
400.0 DELETED  mockllm-aa-1  0/1  Terminating  0  10m
"""


def test_timeline(tmp_path):
    assert m.timeline(LOG.splitlines(), "mockllm") == [(100.0, 1), (140.0, 2), (300.0, 1), (400.0, 0)]
    p = tmp_path / "w.log"
    p.write_text(LOG)
    assert m.main([str(p), "--deployment", "mockllm", "--expect-zero"]) == 0
    assert m.main([str(p), "--deployment", "mockllm", "--max-replicas", "1"]) == 1
