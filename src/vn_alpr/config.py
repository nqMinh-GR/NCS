"""Load and validate the central project config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ALLOWED_RUN_MODES = {"auto", "train", "edge", "report"}
ALLOWED_PROFILES = {"smoke", "screen", "full", "edge_only", "report_only"}

REQUIRED = {
    "project": str,
    "run_mode": str,
    "profile": str,
    "seed": int,
    "commercial_only": bool,
    "run_root": str,
    "max_download_gb": (int, float),
    "allow_network": bool,
    "allow_unverified_weights": bool,
    "allow_pseudo_labels": bool,
    "enable_optional_older_baselines": bool,
    "onnx_opset": int,
    "detectors_screen": list,
    "detectors_full": list,
    "ocrs_screen": list,
    "ocrs_full": list,
}


class ConfigError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def validate_project(cfg: dict[str, Any]) -> None:
    for key, kind in REQUIRED.items():
        if key not in cfg:
            raise ConfigError(f"missing config key: {key}")
        if not isinstance(cfg[key], kind):
            raise ConfigError(f"{key} has type {type(cfg[key]).__name__}, expected {kind}")
    if cfg["run_mode"] not in ALLOWED_RUN_MODES:
        raise ConfigError(f"run_mode must be one of {sorted(ALLOWED_RUN_MODES)}")
    if cfg["profile"] not in ALLOWED_PROFILES:
        raise ConfigError(f"profile must be one of {sorted(ALLOWED_PROFILES)}")
    if cfg["onnx_opset"] < 11 or cfg["onnx_opset"] > 20:
        raise ConfigError("onnx_opset must stay in 11..20; do not pin the newest opset")
    if cfg["max_download_gb"] <= 0:
        raise ConfigError("max_download_gb must be positive")
    for section in ("detection", "ocr", "training", "quantization", "edge", "benchmark"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"missing mapping: {section}")
    if cfg["quantization"]["calibration_split"] != "train":
        raise ConfigError("calibration_split must be train")
    if cfg["edge"]["change_power_mode_automatically"] is not False:
        raise ConfigError("change_power_mode_automatically must stay false")


def load_project(path: Path | None = None) -> dict[str, Any]:
    path = path or (repo_root() / "configs" / "project.yaml")
    cfg = load_yaml(path)
    validate_project(cfg)
    return cfg


def train_hparams(cfg: dict[str, Any]) -> dict[str, Any]:
    training = cfg["training"]
    return {
        "det_batch_size": int(training.get("det_batch_size", 4)),
        "ocr_batch_size": int(training.get("ocr_batch_size", 16)),
        "grad_accum_steps": int(training.get("grad_accum_steps", 1)),
        "num_workers": int(training.get("num_workers", 2)),
        "learning_rate": training.get("learning_rate"),
        "det_input_size": int(training.get("det_input_size", cfg["detection"].get("default_input", 640))),
        "force_det_input": bool(training.get("force_det_input", False)),
        "use_gpu": bool(training.get("use_gpu", False)),
        "max_images": training.get("max_images"),
        "det_epochs": int(training["det_max_epochs"]),
        "ocr_epochs": int(training["ocr_max_epochs"]),
        "seed": int(cfg["seed"]),
        "python_bins": dict(cfg.get("python_bins") or {}),
    }


def apply_train_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the notebook knobs after the profile, so the edited cell wins."""
    out = copy.deepcopy(cfg)
    if not overrides:
        _validate_train_numbers(out)
        return out
    training = out["training"]
    renamed = {
        "det_epochs": "det_max_epochs",
        "ocr_epochs": "ocr_max_epochs",
        "det_batch_size": "det_batch_size",
        "ocr_batch_size": "ocr_batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "num_workers": "num_workers",
        "det_input_size": "det_input_size",
        "max_images": "max_images",
    }
    for source, dest in renamed.items():
        if source in overrides and overrides[source] is not None:
            training[dest] = overrides[source]
    if "learning_rate" in overrides:
        training["learning_rate"] = overrides["learning_rate"]
    if "force_det_input" in overrides:
        training["force_det_input"] = bool(overrides["force_det_input"])
    if "use_gpu" in overrides:
        training["use_gpu"] = bool(overrides["use_gpu"])
    if overrides.get("seed") is not None:
        out["seed"] = int(overrides["seed"])
    if overrides.get("python_bins"):
        out["python_bins"] = {str(key): str(value) for key, value in overrides["python_bins"].items()}
    _validate_train_numbers(out)
    return out


def _validate_train_numbers(cfg: dict[str, Any]) -> None:
    training = cfg["training"]
    positive = ("det_batch_size", "ocr_batch_size", "grad_accum_steps", "det_max_epochs", "ocr_max_epochs", "det_input_size")
    for key in positive:
        if key not in training or training[key] is None:
            continue
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{key} must be an integer >= 1")
    workers = training.get("num_workers")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 0):
        raise ConfigError("num_workers must be an integer >= 0")
    learning_rate = training.get("learning_rate")
    if learning_rate is not None and (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0):
        raise ConfigError("learning_rate must be null or a number > 0")
    max_images = training.get("max_images")
    if max_images is not None and (isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1):
        raise ConfigError("max_images must be null or an integer >= 1")
    if int(training.get("det_input_size", 640)) % 32 != 0:
        raise ConfigError("det_input_size must be a multiple of 32")


def apply_profile(cfg: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    """Return a copy whose budgets match the selected profile."""
    out = copy.deepcopy(cfg)
    profile = profile or out["profile"]
    if profile not in ALLOWED_PROFILES:
        raise ConfigError(f"unknown profile {profile}")
    out["profile"] = profile
    if profile == "smoke":
        out["training"]["det_max_epochs"] = out["training"]["smoke_det_epochs"]
        out["training"]["ocr_max_epochs"] = out["training"]["smoke_ocr_epochs"]
        out["training"]["patience"] = 1
        out["quantization"]["calibration_samples"] = min(8, out["quantization"]["calibration_samples"])
        bench = out["benchmark"]
        bench["warmup_iterations"] = bench["smoke_warmup_iterations"]
        bench["min_timed_iterations"] = bench["smoke_min_timed_iterations"]
        bench["min_timed_seconds"] = bench["smoke_min_timed_seconds"]
        bench["repeats"] = bench["smoke_repeats"]
        bench["sustained_seconds"] = bench["smoke_sustained_seconds"]
        out["smoke_limits_applied"] = True
    else:
        out["smoke_limits_applied"] = False
    return out


def selected_ids(cfg: dict[str, Any], kind: str) -> list[str]:
    profile = cfg["profile"]
    if kind == "detector":
        if profile == "full":
            return list(cfg["detectors_full"])
        if profile in {"screen", "smoke"}:
            return list(cfg["detectors_screen"])
        return []
    if kind == "ocr":
        if profile == "full":
            return list(cfg["ocrs_full"])
        if profile in {"screen", "smoke"}:
            return list(cfg["ocrs_screen"])
        return []
    raise ConfigError(kind)
