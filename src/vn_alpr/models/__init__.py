"""Load the model registry and decide what commercial_only is allowed to run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vn_alpr.config import repo_root


def load_models(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (repo_root() / "configs" / "models.yaml")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return list(data["models"])


def by_id(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["model_id"]: item for item in models}


def selection_reason(model: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    if model.get("commercial_status") == "restricted":
        return False, "restricted"
    if model.get("optional") and not cfg.get("enable_optional_older_baselines"):
        return False, "optional_disabled"
    if not model.get("enabled", False):
        return False, "disabled"
    scratch = model.get("commercial_status") == "unresolved" and model.get("train_from_scratch")
    if cfg.get("commercial_only") and model.get("commercial_status") != "eligible" and not scratch:
        return False, f"commercial_status={model.get('commercial_status')}"
    if model.get("commercial_status") == "unresolved" and not cfg.get("allow_unverified_weights") and not scratch:
        return False, "weights_unresolved"
    if scratch:
        return True, "train_from_scratch_no_pretrained_weights"
    return True, "selected"


def active_models(cfg: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    from vn_alpr.config import selected_ids

    catalog = by_id(load_models())
    chosen = []
    for model_id in selected_ids(cfg, kind):
        model = catalog.get(model_id)
        if model is None:
            continue
        ok, _reason = selection_reason(model, cfg)
        if ok and model.get("task") == ("detection" if kind == "detector" else "ocr"):
            chosen.append(model)
    return chosen


def write_registry_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": load_models()}, indent=2), encoding="utf-8")


def strip_yaml_key_block(text: str, key: str) -> str:
    """Drop a top-level YAML key and its indented children. Used to remove teacher_model."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    key_indent = 0
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not skipping:
            if stripped.startswith(f"{key}:") and not stripped.startswith(f"{key}: "):
                # key: with nested block, or key: value on one line
                if stripped.strip() == f"{key}:" or stripped.startswith(f"{key}:\n") or stripped.rstrip().endswith(":"):
                    if ":" in stripped and stripped.split(":", 1)[1].strip() == "":
                        skipping = True
                        key_indent = indent
                        continue
            if stripped.startswith(f"{key}:"):
                continue
            out.append(line)
            continue
        if stripped.strip() == "":
            continue
        if indent <= key_indent:
            skipping = False
            if stripped.startswith(f"{key}:"):
                continue
            out.append(line)
    return "".join(out)


def zero_distill_loss(text: str) -> str:
    return text.replace("loss_distill: 5", "loss_distill: 0")
