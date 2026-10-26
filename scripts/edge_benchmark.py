#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.benchmark import latency_summary
from vn_alpr.config import load_project, repo_root
from vn_alpr.edge.tensorrt import preflight
from vn_alpr.reporting import write_table


def main() -> int:
    check = preflight()
    cfg = load_project()
    root = repo_root() / cfg["run_root"]
    if check["status"] != "ready":
        write_table(
            root / "jetson_benchmark.csv",
            [
                {
                    "run_id": "edge",
                    "model_id": None,
                    "role": "EDGE_TARGET",
                    "status": "pending_on_device",
                    "reason": check["reason"],
                    "p50_ms": None,
                    "power_w": None,
                    "accuracy": None,
                }
            ],
        )
        print(json.dumps(check, indent=2))
        return 0
    samples = json.loads((root / "timed_ms.json").read_text(encoding="utf-8"))
    print(json.dumps(latency_summary(samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
