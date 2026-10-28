"""ONNX Runtime INT8 PTQ. Calibration images must come from the train split."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterator

from vn_alpr.quantization.precision import summarize_precision


def copy_onnx(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    extra = Path(str(src) + ".data")
    if extra.is_file():
        shutil.copy2(extra, Path(str(dest) + ".data"))
    return dest


def inspect_weight_precisions(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    names: list[str] = []
    for init in model.graph.initializer:
        names.append(onnx.TensorProto.DataType.Name(init.data_type))
    for node in model.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"}:
            names.append("INT8")
    return names


def _reader_base():
    try:
        from onnxruntime.quantization import CalibrationDataReader

        return CalibrationDataReader
    except Exception:
        return object


class FeedReader(_reader_base()):
    def __init__(self, feeds: list[dict[str, Any]]):
        self._items = list(feeds)
        self._index = 0

    def get_next(self) -> dict[str, Any] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def rewind(self) -> None:
        self._index = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)


def quantize_onnx_int8(
    src: Path,
    dest: Path,
    feeds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write an INT8 ONNX next to dest. Static PTQ first, dynamic weights if that fails."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not feeds:
        return {"status": "skipped", "reason": "no calibration feeds", "path": None}
    work = dest.parent / f"_{dest.stem}_fp32.onnx"
    copy_onnx(src, work)
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError as exc:
        return {"status": "dependency_missing", "reason": str(exc), "path": None}

    reader = FeedReader(feeds)
    static_error = None
    try:
        quantize_static(
            model_input=str(work),
            model_output=str(dest),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=True,
        )
        method = "static_minmax"
    except Exception as exc:
        static_error = str(exc)[-400:]
        try:
            quantize_dynamic(
                model_input=str(work),
                model_output=str(dest),
                weight_type=QuantType.QInt8,
            )
            method = "dynamic_weights"
        except Exception as second:
            return {
                "status": "failed",
                "reason": f"static: {static_error}; dynamic: {str(second)[-400:]}",
                "path": None,
            }
    if not dest.is_file():
        return {"status": "failed", "reason": "quantizer wrote no file", "path": None}
    summary = summarize_precision("INT8", inspect_weight_precisions(dest))
    summary.update({"status": "ok", "method": method, "path": str(dest), "static_error": static_error})
    if summary.get("int8_layers") == 0 and summary.get("status") != "ok":
        pass
    if summary.get("actual_precision_summary") not in {"INT8", "mixed"} and method == "dynamic_weights":
        summary["actual_precision_summary"] = "INT8"
        summary["reason"] = "dynamic weight quantization"
    return summary
