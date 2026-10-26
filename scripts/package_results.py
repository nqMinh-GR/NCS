#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.packaging import build_results_zip


def main() -> int:
    cfg = load_project()
    root = repo_root() / cfg["run_root"]
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / "vn_alpr_results.zip"
    info = build_results_zip(root, destination)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
