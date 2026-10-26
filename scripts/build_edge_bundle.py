#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.edge import build_edge_bundle


def main() -> int:
    cfg = load_project()
    root = repo_root() / cfg["run_root"]
    onnx = list(root.rglob("*.onnx"))
    document = build_edge_bundle(root, root / "edge_bundle", onnx)
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
