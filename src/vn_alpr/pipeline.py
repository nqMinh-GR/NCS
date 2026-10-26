"""Orchestrate a profile. Training runs only through upstream commands when dependencies exist."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from vn_alpr.config import apply_profile, apply_train_overrides, load_project, repo_root, train_hparams
from vn_alpr.data.prepare import prepare_locked_split
from vn_alpr.data.synthetic import make_synthetic_detection
from vn_alpr.edge.tensorrt import preflight
from vn_alpr.hardware import write_probe
from vn_alpr.models import active_models, by_id, load_models, selection_reason, write_registry_json
from vn_alpr.models.commands import commands_for
from vn_alpr.quantization.precision import qat_status
from vn_alpr.reporting import write_summary, write_table


def _stack_ready(model: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    stack = model.get("stack")
    module = {"torch": "torch", "paddle": "paddleocr", "keras": "fast_plate_ocr"}.get(stack or "")
    if module is None:
        return False, "no runtime"
    python = (cfg.get("python_bins") or {}).get(stack or "")
    if python:
        if not Path(python).exists():
            return False, f"dependency_missing: {python} does not exist"
        probe = subprocess.run([python, "-c", f"import {module}"], check=False, capture_output=True, text=True)
        if probe.returncode == 0:
            return True, python
        tail = (probe.stderr or probe.stdout or "import failed").strip()[-300:]
        return False, f"dependency_missing: {module} in {python}: {tail}"
    if importlib.util.find_spec(module) is None:
        install = model.get("pypi") or module
        return False, f"dependency_missing: pip install {install}"
    return True, "present"


def run_profile(
    profile: str | None = None,
    run_mode: str | None = None,
    *,
    start: bool = False,
    config_path: Path | None = None,
    self_test_if_no_data: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_project(config_path)
    if profile:
        cfg["profile"] = profile
    if run_mode:
        cfg["run_mode"] = run_mode
    cfg = apply_train_overrides(apply_profile(cfg, cfg["profile"]), overrides)
    hparams = train_hparams(cfg)
    root = repo_root()
    run_dir = (root / cfg["run_root"]).resolve()
    if not start:
        return {"status": "not_started", "profile": cfg["profile"], "run_dir": str(run_dir)}

    run_dir.mkdir(parents=True, exist_ok=True)
    from vn_alpr.models.runner import find_layout

    layout = find_layout(run_dir)
    if layout:
        hparams.update(layout)
    (run_dir / "train_hparams.json").write_text(json.dumps({k: v for k, v in hparams.items() if k != "python_bins"}, indent=2), encoding="utf-8")
    hardware = write_probe(run_dir)
    write_registry_json(run_dir / "model_registry.json")
    catalog = by_id(load_models())
    rows: list[dict[str, Any]] = []
    commands: dict[str, Any] = {}

    for kind in ("detector", "ocr"):
        wanted = cfg["detectors_screen"] if kind == "detector" else cfg["ocrs_screen"]
        if cfg["profile"] == "full":
            wanted = cfg["detectors_full"] if kind == "detector" else cfg["ocrs_full"]
        for model_id in wanted:
            model = catalog.get(model_id)
            if model is None:
                continue
            ok, reason = selection_reason(model, cfg)
            if not ok:
                rows.append(_row(model_id, "skipped", reason, hardware, hparams, kind))
                continue
            ready, ready_reason = _stack_ready(model, cfg)
            epochs = hparams["det_epochs"] if kind == "detector" else hparams["ocr_epochs"]
            spec = commands_for(model, run_dir / model_id, epochs, cfg["seed"], hparams)
            commands[model_id] = {key: value for key, value in spec.items() if key != "script"}
            if not ready:
                rows.append(_row(model_id, "dependency_missing", ready_reason, hardware, hparams, kind))
                continue
            from vn_alpr.models.runner import attempt

            launched = attempt(model, run_dir, epochs, cfg["seed"], cfg["profile"], hparams)
            rows.append(_row(model_id, launched["status"], launched.get("reason") or launched.get("log"), hardware, hparams, kind))

    self_test = None
    if self_test_if_no_data or cfg["profile"] == "smoke":
        synthetic_dir = run_dir / "self_test"
        samples = make_synthetic_detection(synthetic_dir)
        manifest = prepare_locked_split(
            samples,
            synthetic_dir / "canonical",
            seed=cfg["seed"],
            calibration_n=2,
            self_test=True,
        )
        self_test = {"status": "self_test", "counts": manifest["counts"], "not_a_benchmark": True}

    (run_dir / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    write_table(run_dir / "detector_leaderboard.csv", [row for row in rows if _task(row, catalog) == "detection"])
    write_table(run_dir / "ocr_leaderboard.csv", [row for row in rows if _task(row, catalog) == "ocr"])
    write_table(run_dir / "combined_leaderboard.csv", [])
    edge = preflight()
    edge_rows = []
    for row in rows:
        edge_rows.append(
            {
                **row,
                "role": "EDGE_TARGET",
                "status": "pending_on_device",
                "reason": edge["reason"],
                "p50_ms": None,
                "power_w": None,
                "accuracy": None,
            }
        )
    write_table(run_dir / "jetson_benchmark.csv", edge_rows)
    write_summary(run_dir / "recommendation.md", rows)
    qat = {model["model_id"]: qat_status(model["family"]) for model in active_models(cfg, "detector")}
    (run_dir / "qat_support.json").write_text(json.dumps(qat, indent=2), encoding="utf-8")
    report = {
        "status": "implemented",
        "profile": cfg["profile"],
        "run_mode": cfg["run_mode"],
        "hardware_role": hardware["role"],
        "benchmark_on_target": hardware["benchmark_on_target"],
        "train_executed": any(row["status"] == "ok" for row in rows),
        "edge_executed": False,
        "self_test": self_test,
        "rows": rows,
        "run_dir": str(run_dir),
    }
    (run_dir / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _task(row: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    model = catalog.get(row["model_id"], {})
    return model.get("task", "")


def _row(
    model_id: str,
    status: str,
    reason: str,
    hardware: dict[str, Any],
    hparams: dict[str, Any] | None = None,
    kind: str = "detector",
) -> dict[str, Any]:
    batch = 1
    if hparams:
        batch = hparams["ocr_batch_size"] if kind == "ocr" else hparams["det_batch_size"]
    return {
        "run_id": model_id,
        "model_id": model_id,
        "checkpoint_hash": None,
        "dataset_split_hash": None,
        "protocol": None,
        "backend": None,
        "requested_precision": None,
        "actual_precision_summary": None,
        "input_shape": None,
        "batch": batch,
        "role": hardware["role"],
        "hardware_fingerprint": hardware.get("device_tree_model") or hardware.get("machine"),
        "power_mode": hardware.get("nvpmodel"),
        "accuracy": None,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "throughput": None,
        "peak_memory": None,
        "temperature_c": None,
        "power_w": None,
        "status": status,
        "reason": reason,
        "log_path": None,
        "self_test": False,
        "benchmark_on_target": False,
    }


def launch_upstream(argv: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    from vn_alpr.models.runner import launch_upstream as run

    return run(argv, cwd, log_path)
