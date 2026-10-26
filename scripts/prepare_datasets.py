#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import load_project, repo_root
from vn_alpr.data import catalog_plan
from vn_alpr.data.download import env_api_key, manual_zip, roboflow_status
from vn_alpr.data.prepare import prepare_locked_split
from vn_alpr.data.synthetic import make_synthetic_detection


def main() -> int:
    cfg = load_project()
    plan = catalog_plan(cfg["commercial_only"])
    out = repo_root() / cfg["run_root"] / "dataset_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if "--self-test" in sys.argv:
        root = repo_root() / cfg["run_root"] / "self_test"
        samples = make_synthetic_detection(root)
        manifest = prepare_locked_split(samples, root / "canonical", seed=cfg["seed"], self_test=True)
        print("self_test_counts", manifest["counts"])
    if "--roboflow-list" in sys.argv:
        from vn_alpr.data import load_datasets

        key = env_api_key()
        for item in load_datasets():
            if item.get("provider") == "roboflow" and not item.get("excluded"):
                print(json.dumps(roboflow_status(item, key), ensure_ascii=False))
    manual = None
    for arg in sys.argv[1:]:
        if arg.startswith("--manual="):
            manual = Path(arg.split("=", 1)[1])
    if manual:
        print(manual_zip("manual", manual, repo_root() / cfg["run_root"] / "manual"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
