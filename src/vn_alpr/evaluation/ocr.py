"""OCR metrics. Unreadable rows are excluded only when marked before scoring."""

from __future__ import annotations

from collections import defaultdict

from vn_alpr.textnorm import edit_distance, normalize_plate


def evaluate_ocr(samples: list[dict], charset: str | None = None) -> dict:
    primary = []
    excluded = []
    for sample in samples:
        if sample.get("readable") is False or sample.get("label_verification_status") == "ambiguous":
            excluded.append(sample)
        else:
            primary.append(sample)
    exact_norm = 0
    exact_raw = 0
    edit_sum = 0
    char_sum = 0
    empty_wrong = 0
    confusion: dict[str, int] = defaultdict(int)
    by_layout: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    for sample in primary:
        gt_raw = "" if sample.get("gt_raw") is None else str(sample.get("gt_raw"))
        pred_raw = "" if sample.get("pred_raw") is None else str(sample.get("pred_raw"))
        gt_n = normalize_plate(gt_raw)
        pred_n = normalize_plate(pred_raw)
        if pred_raw == "":
            empty_wrong += 1
        ok = gt_n == pred_n and gt_n != ""
        exact_norm += int(ok)
        exact_raw += int(gt_raw == pred_raw and gt_raw != "")
        edit_sum += edit_distance(gt_n, pred_n)
        char_sum += max(len(gt_n), 1)
        layout = sample.get("layout") or "unknown"
        source = sample.get("source") or "unknown"
        by_layout[layout]["n"] += 1
        by_layout[layout]["ok"] += int(ok)
        by_source[source]["n"] += 1
        by_source[source]["ok"] += int(ok)
        if not ok:
            width = max(len(gt_n), len(pred_n))
            for index in range(width):
                gch = gt_n[index] if index < len(gt_n) else ""
                pch = pred_n[index] if index < len(pred_n) else ""
                if gch != pch:
                    confusion[f"{gch}->{pch}"] += 1
    n = len(primary)
    return {
        "n": n,
        "excluded_unreadable_or_ambiguous": len(excluded),
        "coverage": n / (n + len(excluded)) if (n + len(excluded)) else None,
        "exact_normalized": exact_norm / n if n else 0.0,
        "exact_raw": exact_raw / n if n else 0.0,
        "cer": edit_sum / char_sum if char_sum else 0.0,
        "empty_predictions": empty_wrong,
        "by_layout": {k: v["ok"] / v["n"] if v["n"] else None for k, v in by_layout.items()},
        "by_source": {k: v["ok"] / v["n"] if v["n"] else None for k, v in by_source.items()},
        "confusion_top": dict(sorted(confusion.items(), key=lambda item: item[1], reverse=True)[:30]),
        "charset_checked": charset,
        "primary_metric": "exact_normalized",
    }
