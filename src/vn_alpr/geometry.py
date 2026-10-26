"""Boxes are xywh pixels unless a function name says otherwise."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    return x, y, x + w, y + h


def xyxy_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1, y1, x2 - x1, y2 - y1


def box_valid(box: Sequence[float]) -> bool:
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return False
    _x, _y, w, h = values
    return w > 0 and h > 0


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    if not box_valid(a) or not box_valid(b):
        return 0.0
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def letterbox(src_w: int, src_h: int, dst: int) -> dict[str, float]:
    if src_w <= 0 or src_h <= 0 or dst <= 0:
        raise ValueError("letterbox sizes must be positive")
    scale = min(dst / src_w, dst / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    return {
        "scale": scale,
        "pad_x": (dst - new_w) / 2.0,
        "pad_y": (dst - new_h) / 2.0,
        "dst": float(dst),
    }


def xywh_to_letterbox(box: Sequence[float], params: dict[str, float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    scale = params["scale"]
    return x * scale + params["pad_x"], y * scale + params["pad_y"], w * scale, h * scale


def letterbox_xywh_to_original(box: Sequence[float], params: dict[str, float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    scale = params["scale"]
    if scale == 0:
        raise ValueError("scale is 0")
    return (x - params["pad_x"]) / scale, (y - params["pad_y"]) / scale, w / scale, h / scale


def greedy_match(
    gt: Sequence[Sequence[float]],
    preds: Sequence[Sequence[float]],
    iou_thr: float,
) -> list[tuple[int, int, float]]:
    """One-to-one pairs (gt_index, pred_index, iou), highest IoU first."""
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou_xywh(g, p)
            if score >= iou_thr:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, score))
    return pairs


def coco_size_name(area: float) -> str:
    """COCO area buckets: small < 32^2, medium < 96^2, otherwise large."""
    if area < 32 * 32:
        return "coco_small"
    if area < 96 * 96:
        return "coco_medium"
    return "coco_large"


def plate_height_bucket(height: float) -> str:
    """Project buckets on plate height in pixels, separate from COCO area."""
    if height < 32:
        return "plate_h_lt32"
    if height < 64:
        return "plate_h_lt64"
    return "plate_h_ge64"


def mean(values: Iterable[float]) -> float | None:
    data = list(values)
    if not data:
        return None
    return sum(data) / len(data)
