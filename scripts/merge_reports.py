#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.edge import import_edge_results
from vn_alpr.reporting import write_summary, write_table


def main() -> int:
    cfg = load_project()
    root = repo_root() / cfg["run_root"]
    incoming = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "edge_results.json"
    imported = import_edge_results(incoming)
    rows = imported.get("rows") or []
    if rows:
        write_table(root / "jetson_benchmark.csv", rows)
    host = []
    board = root / "detector_leaderboard.json"
    if board.exists():
        host = json.loads(board.read_text(encoding="utf-8"))
    write_summary(root / "recommendation.md", host + rows)
    print(json.dumps({"status": imported["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
