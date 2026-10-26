#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.edge.tensorrt import build_engine, preflight


def main() -> int:
    check = preflight()
    print(json.dumps(check, indent=2))
    if check["status"] != "ready":
        return 0
    cfg = load_project()
    root = repo_root() / cfg["run_root"] / "edge_bundle"
    for onnx in root.glob("*.onnx"):
        result = build_engine(onnx, onnx.with_suffix(".fp16.engine"), "fp16")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
