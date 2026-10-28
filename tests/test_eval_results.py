from pathlib import Path
import csv
import zipfile

import numpy as np
from PIL import Image

from vn_alpr.eval_results import (
    crop_xywh,
    decode_rfdetr_boxes,
    decode_slots,
    decode_xyxy_boxes,
    discover_onnx,
    extract_results_zip,
    load_detection_split,
    load_ocr_rows,
    load_plate_config,
)


def test_decode_slots_strips_pad():
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    logits = np.full((10, 37), -10.0, dtype=np.float32)
    logits[0, alphabet.index("5")] = 8
    logits[1, alphabet.index("1")] = 8
    logits[2, alphabet.index("A")] = 8
    for slot in range(3, 10):
        logits[slot, alphabet.index("_")] = 8
    assert decode_slots(logits, alphabet, "_") == "51A"
    assert decode_slots(logits[None], alphabet, "_") == "51A"


def test_discover_onnx_skips_edge_bundle(tmp_path: Path):
    (tmp_path / "cct_s_v2" / "cct").mkdir(parents=True)
    (tmp_path / "edge_bundle").mkdir()
    (tmp_path / "int8").mkdir()
    (tmp_path / "cct_s_v2" / "cct" / "best.onnx").write_bytes(b"onnx")
    (tmp_path / "edge_bundle" / "best.onnx").write_bytes(b"dup")
    (tmp_path / "int8" / "cct_s_v2.onnx").write_bytes(b"q")
    found = discover_onnx(tmp_path)
    assert [row["model_id"] for row in found] == ["cct_s_v2"]
    assert found[0]["kind"] == "ocr"


def test_extract_results_and_ocr_csv(tmp_path: Path):
    tree = tmp_path / "pack"
    images = tree / "canonical" / "ocr_images"
    images.mkdir(parents=True)
    Image.new("RGB", (32, 16), (10, 20, 30)).save(images / "val_0.jpg")
    csv_path = tree / "canonical" / "val.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        writer.writerow({"image_path": "ocr_images/val_0.jpg", "plate_text": "51A12345"})
        writer.writerow({"image_path": "ocr_images/missing.jpg", "plate_text": "NONE"})
    zipped = tmp_path / "vn_alpr_results.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        for path in tree.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(tree)))
    dest = tmp_path / "out"
    root = extract_results_zip(zipped, dest)
    rows = load_ocr_rows(root / "canonical", "val")
    assert len(rows) == 1
    assert rows[0]["gt_raw"] == "51A12345"
    cfg = load_plate_config(root)
    assert "A" in cfg["alphabet"]


def test_load_detection_split_reads_rfdetr_data(tmp_path: Path):
    import json

    canonical = tmp_path / "canonical"
    image_dir = canonical / "rfdetr_data" / "valid"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (40, 30), (1, 2, 3)).save(image_dir / "plate.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "plate.jpg", "width": 40, "height": 30, "sample_id": "s1", "source": "t"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 5, 10, 8], "area": 80, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "license_plate"}],
    }
    (canonical / "instances_val.json").write_text(json.dumps(coco), encoding="utf-8")
    (canonical / "ocr.jsonl").write_text(json.dumps({"sample_id": "s1", "text_normalized": "51A1"}) + "\n", encoding="utf-8")
    rows = load_detection_split(canonical, "val")
    assert len(rows) == 1
    assert rows[0]["gt"][0]["bbox"] == [4.0, 5.0, 10.0, 8.0]
    assert rows[0]["gt"][0]["text"] == "51A1"


def test_decode_detector_boxes():
    params = {"scale": 1.0, "pad_x": 0.0, "pad_y": 0.0, "dst": 100.0}
    dets = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    labels = np.array([[[4.0, 0.0]]], dtype=np.float32)
    boxes = decode_rfdetr_boxes(dets, labels, 100, 100, params, 0.5)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]["bbox"]
    assert abs(x - 40) < 1 and abs(w - 20) < 1 and abs(h - 40) < 1
    xyxy = decode_xyxy_boxes([[[10, 20, 30, 50]]], [[0.9]], 0.5)
    assert xyxy[0]["bbox"] == [10.0, 20.0, 20.0, 30.0]
    crop = crop_xywh(np.zeros((100, 80, 3), dtype=np.uint8), [10, 10, 20, 20], pad=0.0)
    assert crop.shape[0] == 20 and crop.shape[1] == 20
