#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vn_alpr.config import apply_profile, load_project, repo_root
from vn_alpr.models import active_models
from vn_alpr.models.commands import commands_for


def main() -> int:
    cfg = apply_profile(load_project(), "screen")
    run = repo_root() / cfg["run_root"] / "export_commands.json"
    commands = {}
    for kind in ("detector", "ocr"):
        for model in active_models(cfg, kind):
            epochs = cfg["training"]["det_max_epochs"] if kind == "detector" else cfg["training"]["ocr_max_epochs"]
            spec = commands_for(model, run.parent / model["model_id"], epochs, cfg["seed"])
            commands[model["model_id"]] = spec.get("export_argv") or spec.get("train_argv")
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
