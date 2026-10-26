"""TensorRT engine builds stay on the board that will run them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from vn_alpr.hardware import probe
from vn_alpr.quantization.precision import summarize_precision


def preflight() -> dict[str, Any]:
    hardware = probe()
    trtexec = shutil.which("trtexec")
    return {
        "benchmark_on_target": hardware["benchmark_on_target"],
        "machine": hardware["machine"],
        "device_tree_model": hardware["device_tree_model"],
        "tensorrt": hardware["tensorrt"],
        "trtexec": trtexec,
        "change_power_mode_automatically": False,
        "status": "ready" if hardware["benchmark_on_target"] and trtexec and hardware["tensorrt"] else "pending_on_device",
        "reason": None
        if hardware["benchmark_on_target"] and trtexec
        else "Engine builds run on the Jetson. A server GPU engine is not reusable here.",
    }


def build_engine(onnx: Path, engine: Path, precision: str, workspace_mib: int = 1024) -> dict[str, Any]:
    check = preflight()
    if check["status"] != "ready":
        return {
            "status": "pending_on_device",
            "requested_precision": precision,
            "actual_precision_summary": None,
            "engine": None,
            "reason": check["reason"],
            "latency_ms": None,
            "power_w": None,
        }
    if precision not in {"fp16", "fp32", "int8"}:
        raise ValueError(precision)
    argv = [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--workspace={workspace_mib}",
        "--profilingVerbosity=detailed",
    ]
    if precision == "fp16":
        argv.append("--fp16")
    elif precision == "fp32":
        pass
    else:
        return {
            "status": "unsupported",
            "requested_precision": "int8",
            "actual_precision_summary": None,
            "reason": "INT8 is not implied by --fp16. Supply an explicit Q/DQ ONNX and a calibration cache built on this TensorRT version before building.",
            "engine": None,
            "latency_ms": None,
        }
    log_path = engine.with_suffix(".log")
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return {
            "status": "failed",
            "requested_precision": precision,
            "actual_precision_summary": None,
            "log": str(log_path),
            "reason": f"trtexec exit {completed.returncode}",
        }
    layers = parse_layer_precisions(log_path.read_text(encoding="utf-8"))
    summary = summarize_precision(precision, layers or [precision.upper()])
    return {
        "status": "built",
        "engine": str(engine),
        "log": str(log_path),
        "fingerprint": probe()["device_tree_model"],
        **summary,
    }


def parse_layer_precisions(log_text: str) -> list[str]:
    found = []
    for line in log_text.splitlines():
        upper = line.upper()
        if "INT8" in upper and "LAYER" in upper:
            found.append("INT8")
        elif "FP16" in upper and "LAYER" in upper:
            found.append("FP16")
        elif "FP32" in upper and "LAYER" in upper:
            found.append("FP32")
    return found
