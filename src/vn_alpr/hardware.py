"""Read the machine that is actually running. Missing tools become null plus a reason."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(argv: list[str], timeout: int = 8) -> dict[str, Any]:
    if shutil.which(argv[0]) is None and argv[0] not in {"python3", "cat"}:
        return {"ok": False, "reason": f"{argv[0]} not on PATH", "text": None}
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": str(exc), "text": None}
    text = (completed.stdout or "") + (completed.stderr or "")
    return {"ok": completed.returncode == 0, "reason": None if completed.returncode == 0 else f"exit {completed.returncode}", "text": text.strip() or None}


def _read(path: str) -> str | None:
    file = Path(path)
    if not file.exists():
        return None
    return file.read_text(encoding="utf-8", errors="replace").strip()


def probe() -> dict[str, Any]:
    machine = platform.machine()
    model = _read("/proc/device-tree/model")
    if model:
        model = model.replace("\x00", "")
    tegra = _read("/etc/nv_tegra_release")
    os_release = _read("/etc/os-release")
    cuda = _run(["nvcc", "--version"])
    trt = _run(["python3", "-c", "import tensorrt; print(tensorrt.__version__)"])
    smi = _run(["nvidia-smi", "-L"])
    nvp = _run(["nvpmodel", "-q"])
    clocks = _run(["jetson_clocks", "--show"])
    tegrastats = shutil.which("tegrastats")
    mem = {}
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith(("MemTotal", "MemAvailable", "SwapTotal")):
                key, value = line.split(":", 1)
                mem[key] = value.strip()
    else:
        mem["reason"] = "/proc/meminfo unavailable"
    disk = shutil.disk_usage("/")
    jetson_like = machine in {"aarch64", "arm64"} and bool(model and "jetson" in model.lower() or tegra)
    role = "EDGE_TARGET" if jetson_like else "TRAIN_HOST"
    gpu_name = None
    if smi["ok"] and smi["text"]:
        gpu_name = smi["text"].splitlines()[0]
    return {
        "role": role,
        "benchmark_on_target": bool(jetson_like),
        "uname": " ".join(platform.uname()),
        "machine": machine,
        "device_tree_model": model,
        "device_tree_reason": None if model else "no /proc/device-tree/model",
        "l4t": tegra,
        "l4t_reason": None if tegra else "no /etc/nv_tegra_release",
        "os_release": os_release,
        "python": platform.python_version(),
        "cuda": cuda["text"] if cuda["ok"] else None,
        "cuda_reason": None if cuda["ok"] else cuda["reason"],
        "tensorrt": trt["text"] if trt["ok"] else None,
        "tensorrt_reason": None if trt["ok"] else trt["reason"],
        "nvidia_smi": gpu_name,
        "nvidia_smi_reason": None if smi["ok"] else smi["reason"] or "nvidia-smi failed; Jetson often has no nvidia-smi",
        "meminfo": mem,
        "disk_free_bytes": disk.free,
        "nvpmodel": nvp["text"] if nvp["ok"] else None,
        "nvpmodel_reason": None if nvp["ok"] else nvp["reason"],
        "jetson_clocks": clocks["text"] if clocks["ok"] else None,
        "jetson_clocks_reason": None if clocks["ok"] else clocks["reason"],
        "tegrastats": tegrastats,
        "tegrastats_reason": None if tegrastats else "tegrastats not on PATH",
        "power_mode_changed": False,
        "shared_memory_note": "Jetson RAM is unified. Do not report a separate VRAM pool.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_probe(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hardware = probe()
    environment = {
        "python": hardware["python"],
        "role": hardware["role"],
        "benchmark_on_target": hardware["benchmark_on_target"],
        "tensorrt": hardware["tensorrt"],
        "cuda": hardware["cuda"],
    }
    (out_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return hardware
