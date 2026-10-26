#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.pipeline import run_profile


def main() -> int:
    profile = "screen"
    mode = "auto"
    for arg in sys.argv[1:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
    report = run_profile(profile, mode, start=True)
    print(json.dumps({key: report[key] for key in ("status", "profile", "train_executed", "edge_executed", "run_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
