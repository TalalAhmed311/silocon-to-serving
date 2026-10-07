"""prober.py — an authenticated synthetic probe that feeds the regional health signal (C1).

Runs INSIDE each region (a CronJob or a small Deployment next to the gateway): sends a tiny completion through the
gateway with a dedicated probe tenant's API key, every `interval_s`, and publishes success (1/0) and latency to
CloudWatch (namespace S2S/Failover, dimension Region). Route 53 health checks are then CloudWatch-alarm-based, so no
unauthenticated health endpoint ever has to face the internet (infra/aws/failover).

Run (in-region): S2S_PROBE_KEY=... python -m failover.prober --url http://gateway.s2s:9000 --region us-east-1
Hardware: T0 logic; the CloudWatch push needs boto3 and AWS credentials (IRSA in EKS). TODO(run-on: EKS, two regions)
"""
from __future__ import annotations

import argparse
import os
import time

import httpx

from .health import HealthTracker, Probe


def probe_once(client: httpx.Client, url: str, key: str, model: str, timeout_s: float) -> Probe:
    t0 = time.monotonic()
    try:
        r = client.post(f"{url}/v1/completions", headers={"Authorization": f"Bearer {key}"}, timeout=timeout_s,
                        json={"model": model, "prompt": "ping", "max_tokens": 1, "temperature": 0})
        ok = r.status_code == 200 and bool(r.json().get("choices"))
    except (httpx.HTTPError, ValueError):
        ok = False
    return Probe(t=time.time(), ok=ok, latency_s=time.monotonic() - t0)


def publish(cw, region: str, p: Probe, up: bool) -> None:
    dims = [{"Name": "Region", "Value": region}]
    cw.put_metric_data(Namespace="S2S/Failover", MetricData=[
        {"MetricName": "ProbeSuccess", "Dimensions": dims, "Value": 1.0 if p.ok else 0.0, "Unit": "Count"},
        {"MetricName": "ProbeLatency", "Dimensions": dims, "Value": p.latency_s, "Unit": "Seconds"},
        {"MetricName": "RegionUp", "Dimensions": dims, "Value": 1.0 if up else 0.0, "Unit": "Count"}])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--model", default="mock-llama-8b")
    ap.add_argument("--interval-s", type=float, default=10.0)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true", help="print instead of publishing to CloudWatch")
    a = ap.parse_args()
    key = os.environ["S2S_PROBE_KEY"]                       # from Secrets Manager via the pod's env, never the repo
    cw = None
    if not a.dry_run:
        import boto3
        cw = boto3.client("cloudwatch", region_name=a.region)
    tracker = HealthTracker(latency_slo_s=a.timeout_s)
    with httpx.Client() as c:
        while True:
            p = probe_once(c, a.url, key, a.model, a.timeout_s)
            up = tracker.observe(p)
            if cw:
                publish(cw, a.region, p, up)
            else:
                print(f"{p.t:.0f} ok={p.ok} latency={p.latency_s:.3f}s up={up}", flush=True)
            time.sleep(a.interval_s)


if __name__ == "__main__":
    main()
