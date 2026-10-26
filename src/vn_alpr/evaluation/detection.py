"""Detection metrics. Empty predictions stay in the denominator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from vn_alpr.geometry import box_valid, coco_size_name, greedy_match, plate_height_bucket


def _ap(tp_flags: list[bool], n_gt: int) -> float:
    if n_gt <= 0:
        return 0.0
    tp = 0
    fp = 0
    recalls = []
    precisions = []
    for flag in tp_flags:
        if flag:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    if not recalls:
        return 0.0
    ap = 0.0
    for point in range(101):
        target = point / 100
        precision = 0.0
        for recall, prec in zip(recalls, precisions):
            if recall >= target:
                precision = max(precision, prec)
        ap += precision
    return ap / 101


def _match_at_iou(images: list[dict[str, Any]], iou_thr: float) -> tuple[list[bool], int]:
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    for image in images:
        gt = [box["bbox"] for box in image.get("gt", []) if box_valid(box["bbox"])]
        n_gt += len(gt)
        preds = []
        pred_scores = []
        for pred in image.get("preds", []):
            if not box_valid(pred["bbox"]):
                scored.append((float(pred.get("score", 0.0)), False))
                continue
            preds.append(pred["bbox"])
            pred_scores.append(float(pred.get("score", 0.0)))
        pairs = greedy_match(gt, preds, iou_thr)
        matched_preds = {pi for _gi, pi, _score in pairs}
        for index, score in enumerate(pred_scores):
            scored.append((score, index in matched_preds))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [flag for _score, flag in scored], n_gt


def evaluate_detection(
    images: list[dict[str, Any]],
    score_threshold: float = 0.5,
    iou_match: float = 0.5,
) -> dict[str, Any]:
    n_gt = 0
    n_images = len(images)
    tp = fp = fn = invalid = 0
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0})
    for image in images:
        gt = [box for box in image.get("gt", []) if box_valid(box["bbox"])]
        n_gt += len(gt)
        valid_preds = []
        for pred in image.get("preds", []):
            if float(pred.get("score", 0.0)) < score_threshold:
                continue
            if not box_valid(pred["bbox"]):
                invalid += 1
                fp += 1
                continue
            valid_preds.append(pred)
        pairs = greedy_match(
            [box["bbox"] for box in gt],
            [pred["bbox"] for pred in valid_preds],
            iou_match,
        )
        matched_gt = {gi for gi, _pi, _score in pairs}
        tp += len(pairs)
        fp += len(valid_preds) - len(pairs)
        fn += len(gt) - len(matched_gt)
        for index, box in enumerate(gt):
            _x, _y, w, h = box["bbox"]
            names = [coco_size_name(w * h), plate_height_bucket(h)]
            layout = box.get("layout") or "unknown"
            names.append(f"layout_{layout}")
            source = image.get("source") or "unknown"
            names.append(f"source_{source}")
            hit = index in matched_gt
            for name in names:
                buckets[name]["tp" if hit else "fn"] += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0
    aps = []
    for step in range(10):
        thr = 0.5 + 0.05 * step
        flags, gt_count = _match_at_iou(images, thr)
        aps.append(_ap(flags, gt_count))
    flags50, gt50 = _match_at_iou(images, 0.5)
    recall_by = {}
    for name, counts in buckets.items():
        denom = counts["tp"] + counts["fn"]
        recall_by[name] = counts["tp"] / denom if denom else None
    return {
        "images": n_images,
        "gt_plates": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "invalid_preds": invalid,
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "fp_per_image": fp / n_images if n_images else 0.0,
        "score_threshold": score_threshold,
        "iou_match": iou_match,
        "ap50": _ap(flags50, gt50),
        "ap50_95": sum(aps) / len(aps) if aps else 0.0,
        "recall_by": recall_by,
        "coco_size_definition": "area < 32^2 small, < 96^2 medium, else large",
        "plate_height_definition": "<32, <64, otherwise",
    }
