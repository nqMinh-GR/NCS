#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.evaluation import evaluate_chain, evaluate_detection, evaluate_ocr


def main() -> int:
    cfg = load_project()
    root = repo_root() / cfg["run_root"]
    pred_path = root / "predictions.json"
    if not pred_path.exists():
        print(json.dumps({"status": "no_predictions", "reason": "train/export has not written predictions.json"}))
        return 0
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    detection = evaluate_detection(payload.get("detection", []))
    ocr = evaluate_ocr(payload.get("ocr", []))
    chain = evaluate_chain(payload.get("chain", []))
    out = {"detection": detection, "ocr": ocr, "chain": chain, "self_test": payload.get("self_test", False)}
    (root / "evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
