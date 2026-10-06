"""exporter.py — publish s2s_cost_usd_per_million_tokens{tenant=...} for Grafana, computed from Prometheus data.

Run (in-cluster, or with a port-forward to Prometheus):
  PYTHONPATH=platform uv run python -m cost.exporter --prom http://127.0.0.1:9090 --prices prices.yaml --port 9105
prices.yaml (YOU fill in, from your region's price list / bill — no defaults on purpose):
  nodes:
    - {name: g6.xlarge, usd_per_hour: <price>, gpus: 1, spot: false}
Queries: tokens per tenant per replica from gateway_tokens_total (labels tenant, backend), GPU count per backend
from the replicas table in prices.yaml (backend -> node type, gpus).
"""
from __future__ import annotations

import argparse
import time

import httpx
import yaml
from prometheus_client import Gauge, start_http_server

from .model import Node, Replica, tenant_costs


def query(prom: str, q: str) -> list[dict]:
    r = httpx.get(f"{prom}/api/v1/query", params={"query": q}, timeout=10)
    r.raise_for_status()
    return r.json()["data"]["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--port", type=int, default=9105)
    ap.add_argument("--interval", type=float, default=60)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.prices))
    nodes = {n["name"]: Node(**n) for n in cfg["nodes"]}
    g = Gauge("s2s_cost_usd_per_million_tokens", "Cost per 1M tokens by tenant", ["tenant"])
    start_http_server(a.port)
    while True:
        rows = query(a.prom, 'sum by (tenant, backend) (rate(gateway_tokens_total[1h])) * 3600')
        by_backend: dict = {}
        for row in rows:
            by_backend.setdefault(row["metric"]["backend"], {})[row["metric"]["tenant"]] = float(row["value"][1])
        reps = [Replica(b, nodes[cfg["replicas"][b]["node"]], cfg["replicas"][b]["gpus"], toks)
                for b, toks in by_backend.items() if b in cfg.get("replicas", {})]
        for t, c in tenant_costs(reps).items():
            g.labels(t).set(c["usd_per_million"])
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
