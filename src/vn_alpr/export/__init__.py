"""Compare decoded outputs. A missing runtime is a status, not a made-up error."""

from __future__ import annotations

from typing import Any, Sequence

from vn_alpr.geometry import iou_xywh


def text_parity(native: Sequence[str], exported: Sequence[str]) -> dict[str, Any]:
    if len(native) != len(exported):
        raise ValueError("parity sets must be the same length")
    if not native:
        return {"n": 0, "exact": None, "status": "empty"}
    exact = sum(a == b for a, b in zip(native, exported))
    return {"n": len(native), "exact": exact / len(native), "status": "measured"}


def box_parity(native: Sequence[Sequence[float]], exported: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(native) != len(exported):
        raise ValueError("parity sets must be the same length")
    ious = [iou_xywh(a, b) for a, b in zip(native, exported)]
    if not ious:
        return {"n": 0, "mean_iou": None, "min_iou": None, "status": "empty"}
    return {"n": len(ious), "mean_iou": sum(ious) / len(ious), "min_iou": min(ious), "status": "measured"}


def onnx_checker_status(path: str) -> dict[str, Any]:
    try:
        import onnx
    except ImportError:
        return {"status": "dependency_missing", "reason": "pip install onnx==1.23.2", "path": path}
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return {"status": "checked", "path": path, "ir_version": int(model.ir_version)}
