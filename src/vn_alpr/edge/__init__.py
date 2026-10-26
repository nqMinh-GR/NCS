"""Edge package. Engines are not copied from the training GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vn_alpr.dedup import sha256_file
from vn_alpr.secrets import path_is_sensitive


def build_edge_bundle(run_dir: Path, bundle_dir: Path, onnx_files: list[Path]) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for path in onnx_files:
        if not path.exists() or path_is_sensitive(path):
            continue
        target = bundle_dir / path.name
        target.write_bytes(path.read_bytes())
        manifest.append({"file": target.name, "sha256": sha256_file(target)})
    document = {
        "files": manifest,
        "engines_included": False,
        "reason": "TensorRT engines are built on the Jetson and are not portable from the training GPU.",
        "how_to_run": "Open notebooks/02_jetson_quantize_benchmark.ipynb on the box with this bundle unpacked.",
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def import_edge_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "pending_on_device", "latency_ms": None, "power_w": None, "rows": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("benchmark_on_target") is not True:
        return {
            "status": "rejected",
            "reason": "results are not marked benchmark_on_target",
            "latency_ms": None,
            "rows": [],
        }
    return {"status": "imported", "rows": data.get("rows", [])}
