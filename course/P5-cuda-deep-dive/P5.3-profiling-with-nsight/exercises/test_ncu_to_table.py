"""Exercise 3: the ncu CSV → table script, tested on a SYNTHETIC csv (the format, not real numbers)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("ncu_to_table", ROOT / "platform/kernels/bench/ncu_to_table.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

CSV = '''==PROF== Connected to process 1234
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","bin","h","copy(float*)","1","7","Command line profiler metrics","dram__bytes.sum","Mbyte","2,000.00"
"0","1234","bin","h","copy(float*)","1","7","Command line profiler metrics","gpu__time_duration.sum","usecond","10,000.00"
"1","1234","bin","h","sgemm(int)","1","7","Command line profiler metrics","gpu__time_duration.sum","msecond","2.00"
"1","1234","bin","h","sgemm(int)","1","7","Command line profiler metrics","sm__sass_thread_inst_executed_op_ffma_pred_on.sum","","5,000,000,000"
'''


def test_parse_and_table():
    rows = {r["kernel"]: r for r in m.parse(CSV)}
    assert abs(rows["copy(float*)"]["gbs"] - 200.0) < 1e-9          # 2 GB in 10 ms
    assert abs(rows["sgemm(int)"]["tflops"] - 5.0) < 1e-9           # 1e10 FLOPs in 2 ms
    t = m.table(list(rows.values()), peak_gbs=250.0, peak_tflops=10.0)
    assert "| 200.0 | 80% |" in t and "| 5.00 | 50% |" in t
