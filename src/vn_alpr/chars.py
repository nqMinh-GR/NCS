"""Assemble a plate string from character boxes. Model scores are not used."""

from __future__ import annotations

from typing import Any


def _center(box: dict[str, Any]) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0


def _rotate(points: list[tuple[float, float]], angle_rad: float) -> list[tuple[float, float]]:
    import math

    c, s = math.cos(-angle_rad), math.sin(-angle_rad)
    return [(c * x - s * y, s * x + c * y) for x, y in points]


def order_character_boxes(
    boxes: list[dict[str, Any]],
    plate_quad: list[tuple[float, float]] | None = None,
    gap_factor: float = 1.5,
    min_annotation_confidence: float | None = None,
) -> dict[str, Any]:
    """Cluster rows, then read each row left to right. Top row comes first.

    ``confidence`` on a box is an annotation-quality value from the source.
    A missing value is not treated as a model score and is not filtered.
    """
    if min_annotation_confidence is not None:
        kept = []
        for box in boxes:
            conf = box.get("annotation_confidence")
            if conf is None or float(conf) >= min_annotation_confidence:
                kept.append(box)
        boxes = kept
    if not boxes:
        return {
            "text": "",
            "lines": [],
            "status": "ambiguous",
            "reason": "no_character_boxes",
            "label_origin": "derived_character_boxes",
        }

    angle = 0.0
    if plate_quad and len(plate_quad) >= 2:
        import math

        x0, y0 = plate_quad[0]
        x1, y1 = plate_quad[1]
        angle = math.atan2(y1 - y0, x1 - x0)

    centers = _rotate([_center(box) for box in boxes], angle)
    enriched = []
    for box, (cx, cy) in zip(boxes, centers):
        enriched.append({**box, "_cx": cx, "_cy": cy, "_h": float(box["h"])})
    enriched.sort(key=lambda item: item["_cy"])

    rows: list[list[dict[str, Any]]] = []
    for item in enriched:
        placed = False
        for row in rows:
            ref = sum(box["_cy"] for box in row) / len(row)
            ref_h = sum(box["_h"] for box in row) / len(row)
            if abs(item["_cy"] - ref) <= 0.6 * max(ref_h, item["_h"]):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    rows.sort(key=lambda row: sum(box["_cy"] for box in row) / len(row))

    reasons: list[str] = []
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda item: item["_cx"])
        chars = [str(item["char"]) for item in row]
        widths = sorted(float(item["w"]) for item in row)
        median_w = widths[len(widths) // 2]
        for left, right in zip(row, row[1:]):
            gap = right["_cx"] - left["_cx"] - float(left["w"])
            if gap > gap_factor * median_w:
                reasons.append("large_gap")
            if gap < -0.3 * median_w:
                reasons.append("overlap")
        if len(chars) != len(set(id(item) for item in row)):
            reasons.append("duplicate")
        lines.append("".join(chars))

    status = "ambiguous" if reasons else "assembled"
    return {
        "text": "".join(lines),
        "lines": lines,
        "status": status,
        "reason": ",".join(sorted(set(reasons))),
        "label_origin": "derived_character_boxes",
    }
