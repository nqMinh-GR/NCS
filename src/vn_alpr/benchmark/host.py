"""Time ONNX inference on the machine that is actually running.

The numbers are TRAIN_HOST latency. They are not Jetson results.
Missing artifacts stay null; they are never written as 0.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from vn_alpr.benchmark import latency_summary

NULL_STATUSES = {"no_artifact", "dependency_missing", "failed", "oom", "skipped"}


def time_calls(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    min_seconds: float,
    sync: Callable[[], None] | None = None,
) -> list[float]:
    """Return per-call milliseconds. Warmup samples are discarded."""
    for _ in range(max(0, warmup)):
        fn()
    if sync:
        sync()
    samples: list[float] = []
    started = time.perf_counter()
    limit = max(iterations, 1) * 100
    while len(samples) < iterations or (time.perf_counter() - started) < min_seconds:
        tick = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - tick) * 1000.0)
        if len(samples) >= limit:
            break
    return samples


def concrete_shape(raw_shape: Sequence[Any], batch: int, spatial: int = 640) -> list[int]:
    rank = len(raw_shape)
    dims: list[int] = []
    for index, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            dims.append(int(dim))
        elif index == 0:
            dims.append(batch)
        elif rank == 4 and index in {2, 3}:
            dims.append(spatial)
        else:
            dims.append(1)
    return dims


def _numpy_dtype(type_name: str):
    import numpy as np

    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }.get(type_name, np.float32)


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda malloc" in text


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_name(provider: str, hardware: dict[str, Any]) -> str:
    if provider == "CUDAExecutionProvider":
        try:
            import torch

            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        return str(hardware.get("nvidia_smi") or "cuda")
    return str(hardware.get("machine") or "cpu")


def _open_session(path: Path, prefer_cuda: bool):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = list(ort.get_available_providers())
    providers: list[str] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    return session, str(session.get_providers()[0])


def _feeds(session: Any, batch: int):
    import numpy as np

    feeds = {}
    shapes = []
    for item in session.get_inputs():
        raw = list(item.shape)
        fixed = raw[0] if raw and isinstance(raw[0], int) and raw[0] > 0 else None
        if fixed is not None and fixed != batch:
            return None, fixed
        shape = concrete_shape(raw, fixed or batch)
        feeds[item.name] = np.zeros(shape, dtype=_numpy_dtype(item.type))
        shapes.append(shape)
    return feeds, shapes


def _blank_row(model_id: str, path: Path | None, hardware: dict[str, Any], status: str, reason: str, batch: int) -> dict[str, Any]:
    return {
        "run_id": model_id,
        "model_id": model_id,
        "checkpoint_hash": None,
        "dataset_split_hash": None,
        "protocol": "onnx_runtime",
        "backend": None,
        "requested_precision": "fp32",
        "actual_precision_summary": None,
        "input_shape": None,
        "batch": batch,
        "role": "TRAIN_HOST",
        "hardware_fingerprint": hardware.get("nvidia_smi") or hardware.get("machine"),
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
        "log_path": None if path is None else str(path),
        "benchmark_on_target": False,
    }


def _time_one(path: Path, batch: int, cfg: dict[str, Any], hardware: dict[str, Any], model_id: str) -> dict[str, Any]:
    bench = cfg["benchmark"]
    try:
        session, provider = _open_session(path, prefer_cuda=True)
    except Exception as exc:
        if _is_oom(exc):
            return _blank_row(model_id, path, hardware, "oom", str(exc), batch)
        try:
            session, provider = _open_session(path, prefer_cuda=False)
        except Exception as second:
            return _blank_row(model_id, path, hardware, "failed", str(second), batch)
        fallback = str(exc)
    else:
        fallback = None
    feeds, shapes = _feeds(session, batch)
    if feeds is None:
        return _blank_row(
            model_id,
            path,
            hardware,
            "skipped",
            f"ONNX batch is fixed at {shapes}; requested batch {batch} was not run",
            batch,
        )
    sync = _cuda_sync if provider == "CUDAExecutionProvider" else None
    if provider == "CUDAExecutionProvider":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def run_once() -> None:
        session.run(None, feeds)

    try:
        samples: list[float] = []
        repeats = int(bench["repeats"])
        for _ in range(repeats):
            samples.extend(
                time_calls(
                    run_once,
                    warmup=int(bench["warmup_iterations"]),
                    iterations=int(bench["min_timed_iterations"]),
                    min_seconds=float(bench["min_timed_seconds"]),
                    sync=sync,
                )
            )
    except Exception as exc:
        status = "oom" if _is_oom(exc) else "failed"
        return _blank_row(model_id, path, hardware, status, str(exc), batch)
    summary = latency_summary(samples)
    images_per_s = summary["throughput_per_s"] * batch
    peak = None
    if provider == "CUDAExecutionProvider":
        try:
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
        except Exception:
            peak = None
    reason = "TRAIN_HOST latency. Not a Jetson measurement."
    if fallback:
        reason += f" CUDA provider failed, fell back to {provider}: {fallback}"
    row = _blank_row(model_id, path, hardware, "measured", reason, batch)
    row.update(
        {
            "backend": provider,
            "hardware_fingerprint": _device_name(provider, hardware),
            "input_shape": str(shapes[0] if len(shapes) == 1 else shapes),
            "actual_precision_summary": str(next(iter(feeds.values())).dtype),
            "p50_ms": summary["p50_ms"],
            "p95_ms": summary["p95_ms"],
            "p99_ms": summary["p99_ms"],
            "throughput": images_per_s,
            "peak_memory": peak,
            "mean_ms": summary["mean_ms"],
            "std_ms": summary["std_ms"],
            "n": summary["n"],
        }
    )
    return row


def measure_host_latency(run_dir: Path, cfg: dict[str, Any], hardware: dict[str, Any]) -> list[dict[str, Any]]:
    """Time every ONNX file under the run directory. Batch 1, then larger batches when the graph allows it."""
    files = sorted(path for path in run_dir.rglob("*.onnx") if path.is_file() and "self_test" not in path.parts)
    if not files:
        return [_blank_row("none", None, hardware, "no_artifact", "no ONNX file to time", 1)]
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return [
            _blank_row(path.parent.name, path, hardware, "dependency_missing", "pip install onnxruntime==1.31.0", 1)
            for path in files
        ]
    batches = [int(cfg["benchmark"]["batch_size_primary"])]
    for extra in cfg["benchmark"].get("extra_batches_if_fit") or []:
        if int(extra) not in batches:
            batches.append(int(extra))
    rows: list[dict[str, Any]] = []
    for path in files:
        model_id = path.relative_to(run_dir).parts[0] if path.parent != run_dir else path.stem
        for batch in batches:
            row = _time_one(path, batch, cfg, hardware, model_id)
            rows.append(row)
            if row["status"] in {"failed", "oom", "dependency_missing"}:
                break
            if row["status"] == "skipped":
                break
    return rows
