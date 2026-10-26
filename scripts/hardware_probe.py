#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.hardware import write_probe


def main() -> int:
    cfg = load_project()
    out = repo_root() / cfg["run_root"]
    hardware = write_probe(out)
    print(hardware["role"], hardware["machine"], "benchmark_on_target", hardware["benchmark_on_target"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
