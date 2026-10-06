"""specs.py — load and validate gpu_specs.yaml. Imported by examples, exercises, P1.5's calculator and benches.

Validation rules (enforced, so a careless edit can't slip an unsourced number into the course):
  * every entry needs name, source, status
  * status must be "UNVERIFIED" or start with "VERIFIED (" and name a page/table and a date
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parent / "gpu_specs.yaml"
VERIFIED_RE = re.compile(r"^VERIFIED \(.+,.+,\s*\d{4}-\d{2}-\d{2}\)$")


def load(path: Path = PATH) -> list[dict]:
    gpus = yaml.safe_load(path.read_text())["gpus"]
    for g in gpus:
        for k in ("name", "source", "status"):
            if not g.get(k):
                raise ValueError(f"{g.get('name', '?')}: missing {k}")
        if g["status"] != "UNVERIFIED" and not VERIFIED_RE.match(g["status"]):
            raise ValueError(f"{g['name']}: status must be UNVERIFIED or 'VERIFIED (<doc>, <page/table>, YYYY-MM-DD)'")
    return gpus


def get(name: str) -> dict:
    for g in load():
        if g["name"] == name:
            return g
    raise KeyError(name)


def ridge(g: dict, precision: str = "fp16") -> float | None:
    tf = g.get(f"{precision}_dense_tflops")
    return None if tf is None or not g.get("hbm_gbs") else tf * 1e12 / (g["hbm_gbs"] * 1e9)
