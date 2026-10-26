"""Precision labels follow the engine layers, not the requested mode."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_precision(requested: str, layer_precisions: list[str]) -> dict[str, Any]:
    counts = Counter(item.upper() for item in layer_precisions)
    total = sum(counts.values())
    int8 = counts.get("INT8", 0)
    fp16 = counts.get("FP16", 0)
    fp32 = counts.get("FP32", 0)
    if total == 0:
        return {
            "requested_precision": requested,
            "actual_precision_summary": None,
            "int8_layer_fraction": None,
            "status": "unknown",
            "reason": "no layer report",
        }
    fraction = int8 / total
    if requested.upper() == "INT8" and int8 == 0:
        dominant = "FP16" if fp16 >= fp32 else "FP32"
        return {
            "requested_precision": requested,
            "actual_precision_summary": dominant,
            "int8_layer_fraction": 0.0,
            "fp16_layers": fp16,
            "fp32_layers": fp32,
            "int8_layers": 0,
            "status": "not_int8",
            "reason": "no INT8 layers; the engine is not labeled INT8",
        }
    if int8 and (fp16 or fp32):
        actual = "mixed"
        status = "mixed"
    elif int8 == total:
        actual = "INT8"
        status = "int8"
    elif fp16 == total:
        actual = "FP16"
        status = "fp16"
    elif fp32 == total:
        actual = "FP32"
        status = "fp32"
    else:
        actual = "mixed"
        status = "mixed"
    return {
        "requested_precision": requested,
        "actual_precision_summary": actual,
        "int8_layer_fraction": fraction,
        "fp16_layers": fp16,
        "fp32_layers": fp32,
        "int8_layers": int8,
        "status": status,
        "reason": "counted from the layer report",
    }


def quality_gate(
    fp16_value: float,
    candidate_value: float,
    max_drop_pp: float,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    if higher_is_better:
        drop_pp = (fp16_value - candidate_value) * 100
    else:
        drop_pp = (candidate_value - fp16_value) * 100
    return {
        "drop_percentage_points": drop_pp,
        "max_drop_percentage_points": max_drop_pp,
        "passed": drop_pp <= max_drop_pp,
    }


QAT_SUPPORT = {
    "rtdetrv4": "unsupported_qat",
    "rfdetr": "unsupported_qat",
    "dfine": "unsupported_qat",
    "deim": "unsupported_qat",
    "ppocr": "unsupported_qat",
    "fast_plate_ocr": "unsupported_qat",
    "openocr_svtrv2": "unsupported_qat",
}


def qat_status(family: str) -> dict[str, str]:
    status = QAT_SUPPORT.get(family, "unsupported_qat")
    return {
        "status": status,
        "reason": "No checked export path from a quantized-aware training loop was verified for this family. PTQ and FP16 stay available.",
    }
