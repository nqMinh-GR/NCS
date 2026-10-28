from pathlib import Path
import csv
import zipfile

import numpy as np
from PIL import Image

from vn_alpr.eval_results import (
    decode_slots,
    discover_onnx,
    extract_results_zip,
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
