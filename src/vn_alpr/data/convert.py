"""Canonical COCO and OCR rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from vn_alpr.chars import order_character_boxes
from vn_alpr.textnorm import normalize_plate

PLATE_ALIASES = {
    "license_plate",
    "licence_plate",
    "license",
    "plate",
    "bien_so",
    "bien_so_xe",
    "number_plate",
    "licenseplate",
}
ARTIFACTS = {"images", "labels", "image", "label", "0", "1"}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def class_role(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in ARTIFACTS:
        return "artifact"
    if key in PLATE_ALIASES:
        return "plate"
    if len(key) == 1 and key.isalnum():
        return "character"
    return "unknown"


def coco_document(images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "info": {"description": "vn_alpr license_plate"},
        "licenses": [],
        "categories": [{"id": 1, "name": "license_plate", "supercategory": "plate"}],
        "images": images,
        "annotations": annotations,
    }


def write_coco(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def yolo_line_to_xywh(line: str, width: int, height: int) -> tuple[int, tuple[float, float, float, float]] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = (float(v) for v in parts[1:5])
    x = (cx - bw / 2) * width
    y = (cy - bh / 2) * height
    return cls, (x, y, bw * width, bh * height)


def voc_box_to_xywh(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    return xmin, ymin, xmax - xmin, ymax - ymin


def build_detection_coco(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """samples: image_path, width, height, boxes[{name,bbox,source}], sample_id."""
    images = []
    annotations = []
    rejected = []
    ann_id = 1
    for index, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": index,
                "file_name": sample["file_name"],
                "width": sample["width"],
                "height": sample["height"],
                "sample_id": sample["sample_id"],
                "source": sample.get("dataset_id"),
            }
        )
        for box in sample.get("boxes", []):
            role = class_role(str(box.get("name", "")))
            if role != "plate":
                rejected.append(
                    {
                        "sample_id": sample["sample_id"],
                        "name": box.get("name"),
                        "reason": role,
                    }
                )
                continue
            x, y, w, h = box["bbox"]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco_document(images, annotations), rejected


def ocr_row_from_chars(
    sample: dict[str, Any],
    char_boxes: list[dict[str, Any]],
    plate_quad: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    assembled = order_character_boxes(char_boxes, plate_quad=plate_quad)
    raw = assembled["text"]
    return {
        "sample_id": sample["sample_id"],
        "image_path": sample.get("image_path"),
        "crop_path": sample.get("crop_path"),
        "source_family_id": sample.get("source_family_id"),
        "dataset_id": sample.get("dataset_id"),
        "parent_image_id": sample.get("parent_image_id"),
        "bbox_or_quad": sample.get("bbox_or_quad"),
        "text_raw": raw,
        "text_normalized": normalize_plate(raw),
        "layout": "double_line" if len(assembled["lines"]) > 1 else "single_line",
        "line_texts": assembled["lines"],
        "country": sample.get("country", "VN"),
        "readable": assembled["status"] == "assembled" and raw != "",
        "label_origin": "derived_character_boxes",
        "label_verification_status": assembled["status"],
        "split": sample.get("split"),
        "group_id": sample.get("group_id"),
        "reason": assembled["reason"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rfdetr_layout(coco: dict[str, Any], image_dir: Path, dest: Path, split: str) -> Path:
    """Write train/_annotations.coco.json plus image files are referenced, not copied."""
    split_dir = dest / split
    split_dir.mkdir(parents=True, exist_ok=True)
    document = json.loads(json.dumps(coco))
    for image in document["images"]:
        image["file_name"] = str(Path(image["file_name"]).name)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")
    link_dir = split_dir / "_image_dir.txt"
    link_dir.write_text(str(image_dir), encoding="utf-8")
    return split_dir
