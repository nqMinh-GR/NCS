"""Dataset catalog. Versions that were not listed by an API stay null."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vn_alpr.config import repo_root

CHECKED = "2026-10-26"


def load_datasets(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (repo_root() / "configs" / "datasets.yaml")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return list(data["datasets"])


def by_id(datasets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["dataset_id"]: item for item in datasets}


def selectable(item: dict[str, Any], commercial_only: bool) -> tuple[bool, str]:
    if item.get("excluded"):
        return False, item.get("reason", "excluded")
    status = item["commercial_status"]
    if status == "restricted":
        return False, "restricted"
    if commercial_only and status != "eligible":
        return False, f"commercial_status={status}"
    if item.get("quality_status") == "quarantine":
        return False, "quality_quarantine"
    if item.get("role") == "synthetic_train_only":
        return False, "synthetic_train_only"
    if item.get("role") in {"auxiliary", "mixed_auxiliary"}:
        return True, "auxiliary_capped"
    return True, "eligible"


def catalog_plan(commercial_only: bool = True) -> list[dict[str, Any]]:
    rows = []
    for item in load_datasets():
        ok, reason = selectable(item, commercial_only)
        rows.append(
            {
                "dataset_id": item["dataset_id"],
                "commercial_status": item["commercial_status"],
                "access_status": item.get("access_status"),
                "quality_status": item.get("quality_status"),
                "selected_for_default_download": ok
                and item.get("download_method")
                not in {None, "none", "manual_only_until_version_exists", "generate", "local_path", "manual"},
                "reason": reason,
                "version": item.get("version"),
            }
        )
    return rows
