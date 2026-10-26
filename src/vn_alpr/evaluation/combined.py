"""Detection then OCR. Conditional OCR accuracy is reported separately."""

from __future__ import annotations

from typing import Any

from vn_alpr.geometry import box_valid, greedy_match
from vn_alpr.textnorm import normalize_plate


def evaluate_chain(
    images: list[dict[str, Any]],
    score_threshold: float = 0.5,
    iou_match: float = 0.5,
) -> dict[str, Any]:
    gt_total = 0
    chain_ok = 0
    detected = 0
    ocr_ok_given_det = 0
    fp = 0
    for image in images:
        gt = [item for item in image.get("gt", []) if box_valid(item["bbox"])]
        preds = []
        for pred in image.get("preds", []):
            if float(pred.get("score", 0.0)) < score_threshold:
                continue
            if not box_valid(pred["bbox"]):
                fp += 1
                continue
            preds.append(pred)
        gt_total += len(gt)
        pairs = greedy_match(
            [item["bbox"] for item in gt],
            [pred["bbox"] for pred in preds],
            iou_match,
        )
        matched_preds = {pi for _gi, pi, _score in pairs}
        fp += len(preds) - len(matched_preds)
        detected += len(pairs)
        for gi, pi, _score in pairs:
            gt_text = normalize_plate(gt[gi].get("text"))
            pred_text = normalize_plate(preds[pi].get("text"))
            if gt_text and gt_text == pred_text:
                chain_ok += 1
                ocr_ok_given_det += 1
    return {
        "gt_plates": gt_total,
        "chain_exact": chain_ok / gt_total if gt_total else 0.0,
        "detected": detected,
        "ocr_exact_given_detection": ocr_ok_given_det / detected if detected else None,
        "false_positive_plates": fp,
        "note": "chain_exact counts a miss or a wrong read as failure. ocr_exact_given_detection is not the chain score.",
    }
