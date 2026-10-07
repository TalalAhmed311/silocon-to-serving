"""Writes good.safetensors plus six malformed variants into the given directory.

Run: uv run python exercises/03-safetensors-loader/make_test_files.py build/st-test
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/st-test")
out.mkdir(parents=True, exist_ok=True)
good = out / "good.safetensors"
save_file({"a": np.zeros((4, 4), np.float32), "b": np.ones(8, np.float16), "c": np.arange(3, dtype=np.int64)}, str(good))

raw = good.read_bytes()
(n,) = struct.unpack("<Q", raw[:8])
header = json.loads(raw[8:8 + n])
data = raw[8 + n:]


def write(name: str, hdr: dict, body: bytes = data, header_len: int | None = None) -> None:
    h = json.dumps(hdr).encode()
    pad = (8 - len(h) % 8) % 8          # keep the data section 8-aligned like the real writer does
    h += b" " * pad
    (out / name).write_bytes(struct.pack("<Q", header_len if header_len is not None else len(h)) + h + body)


def patched(**changes) -> dict:
    h = json.loads(json.dumps(header))
    for tensor, fields in changes.items():
        h[tensor].update(fields)
    return h


write("bad_header_len.safetensors", header, header_len=1 << 40)
write("bad_dtype.safetensors", patched(a={"dtype": "F31"}))
a_begin, a_end = header["a"]["data_offsets"]
write("bad_past_end.safetensors", patched(a={"data_offsets": [a_begin, len(data) + 100]}))
write("bad_size_mismatch.safetensors", patched(a={"shape": [4, 5]}))
b_begin, b_end = header["b"]["data_offsets"]
write("bad_overlap.safetensors", patched(b={"data_offsets": [a_begin, a_begin + (b_end - b_begin)]}))
c_begin, c_end = header["c"]["data_offsets"]
write("bad_misaligned.safetensors", patched(c={"data_offsets": [c_begin + 1, c_end + 1]}), body=data + b"\0")
print("wrote", sorted(p.name for p in out.iterdir()))
