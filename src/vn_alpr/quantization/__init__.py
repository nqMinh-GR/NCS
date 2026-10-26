"""Calibration manifests stay inside the train split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def write_calibration_manifest(
    path: Path,
    train_ids: Sequence[str],
    sample_ids: Sequence[str],
    protected_ids: Sequence[str],
    meta: dict,
) -> dict:
    protected = set(protected_ids)
    train = set(train_ids)
    bad = [sample_id for sample_id in sample_ids if sample_id in protected or sample_id not in train]
    if bad:
        raise ValueError(f"calibration leaked outside train: {bad[:5]}")
    document = {
        "ids": list(sample_ids),
        "n": len(sample_ids),
        "split": "train",
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document
