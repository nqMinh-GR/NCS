"""Lock a split, write canonical detection and OCR files, and sample calibration from train."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vn_alpr.data.audit import class_histogram, contact_sheet
from vn_alpr.data.convert import build_detection_coco, ocr_row_from_chars, write_coco, write_jsonl
from vn_alpr.dedup import group_records, image_record
from vn_alpr.splits import (
    assign_splits,
    calibration_ids,
    excluded_official_duplicates,
    leakage,
    sample_ids_for_split,
)
from vn_alpr.textnorm import normalize_plate


def records_for_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        if sample.get("file_sha256") and sample.get("pixel_sha256") and "ahash" in sample:
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "file_sha256": sample["file_sha256"],
                    "pixel_sha256": sample["pixel_sha256"],
                    "ahash": int(sample["ahash"]),
                    "video_id": sample.get("video_id"),
                    "plate_identity": sample.get("plate_identity"),
                }
            )
            continue
        records.append(
            image_record(
                sample["sample_id"],
                Path(sample["image_path"]),
                video_id=sample.get("video_id"),
                plate_identity=sample.get("plate_identity"),
            )
        )
    return records


def prepare_locked_split(
    samples: list[dict[str, Any]],
    out_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    official_test_ids: list[str] | None = None,
    calibration_n: int = 1000,
    self_test: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    official_test_ids = official_test_ids or []
    records = records_for_samples(samples)
    grouped = group_records(records)
    groups = grouped["groups"]
    assignment = assign_splits(groups, ratios=ratios, seed=seed, official_test_ids=official_test_ids)
    problems = leakage(groups, assignment)
    if problems:
        raise RuntimeError(f"split leakage: {problems}")
    duplicates = excluded_official_duplicates(groups, assignment, official_test_ids)
    excluded_ids = {row["sample_id"] for row in duplicates}
    sample_group = {}
    for gid, members in groups.items():
        for sample_id in members:
            sample_group[sample_id] = gid
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        sample = dict(sample)
        sample["group_id"] = sample_group[sample["sample_id"]]
        sample["split"] = assignment[sample["group_id"]]
        if sample["sample_id"] in excluded_ids:
            sample["split"] = "excluded_duplicate_of_test"
            continue
        by_split[sample["split"]].append(sample)

    coco_paths = {}
    for split, rows in by_split.items():
        document, rejected = build_detection_coco(rows)
        path = out_dir / f"instances_{split}.json"
        write_coco(path, document)
        coco_paths[split] = str(path)
        (out_dir / f"rejected_{split}.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")

    ocr_rows = []
    for sample in samples:
        if sample["sample_id"] in excluded_ids:
            continue
        chars = sample.get("char_boxes")
        if chars:
            row = ocr_row_from_chars(sample, chars, sample.get("plate_quad"))
        elif sample.get("text_raw"):
            raw = str(sample["text_raw"])
            row = {
                "sample_id": sample["sample_id"],
                "image_path": sample.get("image_path"),
                "crop_path": sample.get("crop_path") or sample.get("image_path"),
                "source_family_id": sample.get("source_family_id"),
                "dataset_id": sample.get("dataset_id"),
                "parent_image_id": sample.get("parent_image_id", sample["sample_id"]),
                "bbox_or_quad": (sample.get("boxes") or [{}])[0].get("bbox"),
                "text_raw": raw,
                "text_normalized": normalize_plate(raw),
                "layout": sample.get("layout") or "unknown",
                "line_texts": sample.get("line_texts") or [raw],
                "country": sample.get("country", "VN"),
                "readable": True,
                "label_origin": sample.get("label_origin", "source"),
                "label_verification_status": "source",
                "split": assignment[sample_group[sample["sample_id"]]],
                "group_id": sample_group[sample["sample_id"]],
            }
        else:
            continue
        row["self_test"] = self_test
        ocr_rows.append(row)
    write_jsonl(out_dir / "ocr.jsonl", ocr_rows)

    train_ids = sample_ids_for_split(groups, assignment, "train")
    train_ids = [sample_id for sample_id in train_ids if sample_id not in excluded_ids]
    calib = calibration_ids(train_ids, calibration_n, seed)
    manifest = {
        "seed": seed,
        "self_test": self_test,
        "groups": groups,
        "assignment": assignment,
        "official_test_duplicates_removed_from_train": duplicates,
        "phash_audit": grouped["phash_audit"],
        "counts": {split: len(rows) for split, rows in by_split.items()},
        "class_histogram": class_histogram(samples),
        "coco": coco_paths,
        "calibration_ids": calib,
        "calibration_from": "train",
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps({"ids": calib, "split": "train", "seed": seed, "n": len(calib)}, indent=2),
        encoding="utf-8",
    )
    images = [Path(sample["image_path"]) for sample in samples if sample.get("image_path")]
    if images:
        contact_sheet(images[:8], out_dir / "contact_sheet.png")
    return manifest
