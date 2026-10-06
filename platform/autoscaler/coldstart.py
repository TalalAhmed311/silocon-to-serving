"""coldstart.py — turn `kubectl get events` + pod conditions into a cold-start phase table (T3).

Run during a scale-from-zero: PYTHONPATH=platform python -m autoscaler.coldstart --namespace s2s --pod <vllm-pod>
Phases: scheduled-wait (Pending → node available), image pull, container start → Ready (weights + warm-up).
Output: | phase | seconds |   (paste into P3.6 exercise 3 and replace the placeholders in animations/p3-request-path.html)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="s2s")
    ap.add_argument("--pod", required=True)
    a = ap.parse_args()
    pod = json.loads(subprocess.check_output(["kubectl", "-n", a.namespace, "get", "pod", a.pod, "-o", "json"]))
    ev = json.loads(subprocess.check_output(["kubectl", "-n", a.namespace, "get", "events", "-o", "json",
                                             "--field-selector", f"involvedObject.name={a.pod}"]))["items"]
    first = lambda reason: min((ts(e.get("eventTime") or e["firstTimestamp"]) for e in ev if e["reason"] == reason), default=None)  # noqa: E731
    created = ts(pod["metadata"]["creationTimestamp"])
    cond = {c["type"]: ts(c["lastTransitionTime"]) for c in pod["status"].get("conditions", [])}
    marks = [("created", created), ("scheduled", first("Scheduled")), ("pulling", first("Pulling")),
             ("pulled", first("Pulled")), ("started", first("Started")), ("ready", cond.get("Ready"))]
    print("| phase | seconds |\n|---|---|")
    for (n1, t1), (n2, t2) in zip(marks, marks[1:]):
        if t1 and t2:
            print(f"| {n1} → {n2} | {(t2 - t1).total_seconds():.0f} |")
    if marks[-1][1]:
        print(f"| **total created → ready** | **{(marks[-1][1] - created).total_seconds():.0f}** |")


if __name__ == "__main__":
    main()
