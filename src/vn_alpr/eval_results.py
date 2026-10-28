"""Run OCR eval, INT8 PTQ, and Colab GPU latency on a packaged vn_alpr_results tree."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from vn_alpr.benchmark.host import _open_session, measure_onnx_files
from vn_alpr.evaluation.ocr import evaluate_ocr
from vn_alpr.quantization.precision import quality_gate
from vn_alpr.quantization.ptq_onnx import quantize_onnx_int8
from vn_alpr.reporting import write_summary, write_table

DRIVE_ZIP_CANDIDATES = (
    Path("/content/drive/MyDrive/vn_alpr/vn_alpr_results.zip"),
    Path("/content/drive/MyDrive/NCS/vn_alpr_results.zip"),
    Path("/content/drive/MyDrive/vn_alpr_results.zip"),
)

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
DEFAULT_PAD = "_"
SKIP_ONNX_PARTS = {"edge_bundle", "int8", "self_test"}


def find_results_zip(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    for path in DRIVE_ZIP_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("vn_alpr_results.zip not found on Drive. Set RESULTS_ZIP.")


def extract_results_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "canonical" / "val.csv"
    if marker.is_file():
        return dest
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    if not marker.is_file():
        nested = [path for path in dest.iterdir() if path.is_dir() and (path / "canonical" / "val.csv").is_file()]
        if len(nested) == 1:
            return nested[0]
        raise FileNotFoundError(f"{dest} is not a vn_alpr_results tree (missing canonical/val.csv)")
    return dest


def discover_onnx(results_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.onnx")):
        if any(part in SKIP_ONNX_PARTS for part in path.parts):
            continue
        if path.name.endswith(".data"):
            continue
        relative = path.relative_to(results_dir)
        model_id = relative.parts[0]
        kind = "ocr" if model_id.startswith(("cct", "svtr", "ppocr")) else "detector"
        found.append({"model_id": model_id, "kind": kind, "path": path})
    return found


def load_plate_config(results_dir: Path) -> dict[str, Any]:
    for path in results_dir.rglob("plate_config.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("alphabet"):
            return data
    return {"alphabet": DEFAULT_ALPHABET, "pad_char": DEFAULT_PAD, "img_height": 64, "img_width": 128}


def load_ocr_rows(canonical: Path, split: str, max_images: int | None = None) -> list[dict[str, Any]]:
    csv_path = canonical / f"{split}.csv"
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = (row.get("image_path") or "").strip()
            text = (row.get("plate_text") or "").strip()
            if not rel or not text:
                continue
            path = canonical / rel
            if not path.is_file():
                continue
            rows.append({"image_path": path, "gt_raw": text, "split": split})
            if max_images is not None and len(rows) >= int(max_images):
                break
    return rows


def decode_slots(logits, alphabet: str, pad_char: str) -> str:
    import numpy as np

    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return ""
    if arr.shape[0] == len(alphabet) and arr.shape[1] != len(alphabet):
        arr = arr.T
    indices = arr.argmax(axis=-1)
    chars = []
    for index in indices:
        if 0 <= int(index) < len(alphabet):
            char = alphabet[int(index)]
            if char != pad_char:
                chars.append(char)
    return "".join(chars)


def _read_rgb(path: Path):
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def preprocess_ocr(image, session, plate_cfg: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    height = int(plate_cfg.get("img_height") or 64)
    width = int(plate_cfg.get("img_width") or 128)
    resized = Image.fromarray(image).resize((width, height), Image.BILINEAR)
    arr = np.asarray(resized)
    item = session.get_inputs()[0]
    name = item.type or ""
    if "uint8" in name:
        batch = np.expand_dims(arr, 0)
    else:
        batch = np.expand_dims(arr.astype("float32") / 255.0, 0)
        if len(item.shape) == 4 and item.shape[1] == 3:
            batch = np.transpose(batch, (0, 3, 1, 2))
    return {item.name: batch}


def preprocess_detector(image, session) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    feeds = {}
    height, width = int(image.shape[0]), int(image.shape[1])
    for item in session.get_inputs():
        raw = list(item.shape)
        kind = item.type or ""
        if "int64" in kind:
            feeds[item.name] = np.array([[height, width]], dtype=np.int64)
            continue
        size = 640
        for dim in raw[1:]:
            if isinstance(dim, int) and dim >= 32:
                size = int(dim)
                break
        rgb = Image.fromarray(image)
        scale = min(size / width, size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = rgb.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
        arr = np.asarray(canvas).astype("float32") / 255.0
        nchw = np.transpose(arr, (2, 0, 1))[None]
        if "rfdetr" in str(getattr(session, "_model_path", "")).lower():
            mean = np.array([0.485, 0.456, 0.406], dtype="float32")[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype="float32")[:, None, None]
            nchw = (nchw - mean) / std
        if raw and isinstance(raw[0], int) and raw[0] > 0 and nchw.shape[0] != raw[0]:
            raise ValueError(f"fixed batch {raw[0]} cannot take {nchw.shape[0]}")
        feeds[item.name] = nchw
    return feeds


def _session(path: Path, prefer_cuda: bool):
    session, provider = _open_session(path, prefer_cuda=prefer_cuda)
    session._model_path = str(path)
    return session, provider


def eval_ocr_model(
    path: Path,
    rows: list[dict[str, Any]],
    plate_cfg: dict[str, Any],
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    session, provider = _session(path, prefer_cuda)
    alphabet = str(plate_cfg.get("alphabet") or DEFAULT_ALPHABET)
    pad = str(plate_cfg.get("pad_char") or DEFAULT_PAD)
    samples = []
    for row in rows:
        image = _read_rgb(row["image_path"])
        feeds = preprocess_ocr(image, session, plate_cfg)
        outputs = session.run(None, feeds)
        pred = decode_slots(outputs[0], alphabet, pad)
        samples.append(
            {
                "gt_raw": row["gt_raw"],
                "pred_raw": pred,
                "readable": True,
                "layout": "unknown",
                "source": "results_zip",
            }
        )
    metrics = evaluate_ocr(samples)
    metrics.update({"backend": provider, "n_images": len(samples), "model_path": str(path)})
    return metrics


def _calibration_feeds(
    path: Path,
    rows: list[dict[str, Any]],
    plate_cfg: dict[str, Any],
    kind: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    session, _provider = _session(path, prefer_cuda=False)
    feeds = []
    for row in rows[:limit]:
        image = _read_rgb(row["image_path"])
        if kind == "ocr":
            feeds.append(preprocess_ocr(image, session, plate_cfg))
        else:
            try:
                feeds.append(preprocess_detector(image, session))
            except Exception:
                continue
    return feeds


def run_int8_eval(
    results_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    hardware: dict[str, Any],
    *,
    split: str = "val",
    max_images: int | None = None,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = results_dir / "canonical"
    models = discover_onnx(results_dir)
    plate_cfg = load_plate_config(results_dir)
    ocr_rows = load_ocr_rows(canonical, split, max_images=max_images)
    calib_n = int((cfg.get("quantization") or {}).get("calibration_samples") or 256)
    train_rows = load_ocr_rows(canonical, "train", max_images=calib_n)
    if not train_rows:
        train_rows = ocr_rows[:calib_n]
    quant_dir = out_dir / "int8"
    quant_dir.mkdir(parents=True, exist_ok=True)

    ocr_metrics: dict[str, Any] = {}
    quant_rows: list[dict[str, Any]] = []
    int8_files: list[tuple[str, Path]] = []
    fp32_files: list[tuple[str, Path]] = []
    for model in models:
        model_id = model["model_id"]
        path = model["path"]
        fp32_files.append((model_id, path))
        print(f"--- quantize {model_id} ---", flush=True)
        feeds = _calibration_feeds(path, train_rows, plate_cfg, model["kind"], calib_n)
        dest = quant_dir / f"{model_id}.onnx"
        quant = quantize_onnx_int8(path, dest, feeds)
        quant["model_id"] = model_id
        quant["kind"] = model["kind"]
        quant_rows.append(quant)
        print(model_id, quant.get("status"), quant.get("method"), quant.get("reason", "")[:180], flush=True)
        if quant.get("status") == "ok" and dest.is_file():
            int8_files.append((model_id, dest))
        if model["kind"] != "ocr":
            continue
        print(f"--- ocr {model_id} {split} ---", flush=True)
        fp32 = eval_ocr_model(path, ocr_rows, plate_cfg, prefer_cuda=prefer_cuda)
        payload = {"fp32": fp32}
        if dest.is_file() and quant.get("status") == "ok":
            try:
                payload["int8"] = eval_ocr_model(dest, ocr_rows, plate_cfg, prefer_cuda=prefer_cuda)
                payload["gate"] = quality_gate(
                    float(fp32["exact_normalized"]),
                    float(payload["int8"]["exact_normalized"]),
                    float((cfg.get("quantization") or {}).get("max_ocr_exact_drop_pp") or 1.0),
                )
            except Exception as exc:
                payload["int8_error"] = str(exc)[-400:]
        ocr_metrics[model_id] = payload
        print(
            model_id,
            "fp32_exact",
            fp32.get("exact_normalized"),
            "int8_exact",
            (payload.get("int8") or {}).get("exact_normalized"),
            flush=True,
        )

    print("--- latency fp32 ---", flush=True)
    fp32_latency = measure_onnx_files(fp32_files, cfg, hardware, requested_precision="fp32")
    print("--- latency int8 ---", flush=True)
    int8_latency = measure_onnx_files(int8_files, cfg, hardware, requested_precision="INT8") if int8_files else []

    det_note = {
        "status": "skipped",
        "reason": "detection scene images are not in vn_alpr_results.zip (raw was excluded). OCR crops in canonical/ocr_images are evaluated.",
        "coco": str(canonical / f"instances_{split}.json"),
        "coco_exists": (canonical / f"instances_{split}.json").is_file(),
    }
    report = {
        "status": "implemented",
        "split": split,
        "results_dir": str(results_dir),
        "ocr_images": len(ocr_rows),
        "models": [{"model_id": m["model_id"], "kind": m["kind"], "path": str(m["path"])} for m in models],
        "quantization": quant_rows,
        "ocr": ocr_metrics,
        "detection": det_note,
        "fp32_latency": fp32_latency,
        "int8_latency": int8_latency,
        "benchmark_on_target": False,
        "role": "TRAIN_HOST",
    }
    (out_dir / "int8_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_table(out_dir / "int8_quantization.csv", [_quant_row(item, hardware) for item in quant_rows])
    write_table(out_dir / "fp32_latency.csv", fp32_latency)
    write_table(out_dir / "int8_latency.csv", int8_latency)
    summary_rows = _summary_rows(ocr_metrics, fp32_latency, int8_latency, hardware)
    write_table(out_dir / "ocr_int8_leaderboard.csv", summary_rows)
    write_summary(out_dir / "recommendation.md", summary_rows, fp32_latency + int8_latency)
    return report


def _quant_row(item: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": item.get("model_id"),
        "model_id": item.get("model_id"),
        "requested_precision": "INT8",
        "actual_precision_summary": item.get("actual_precision_summary"),
        "status": item.get("status"),
        "reason": item.get("reason") or item.get("method"),
        "role": "TRAIN_HOST",
        "hardware_fingerprint": hardware.get("nvidia_smi") or hardware.get("machine"),
        "log_path": item.get("path"),
        "benchmark_on_target": False,
    }


def _summary_rows(
    ocr_metrics: dict[str, Any],
    fp32_latency: list[dict[str, Any]],
    int8_latency: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    lat = {}
    for row in fp32_latency + int8_latency:
        if row.get("batch") != 1:
            continue
        lat[(row.get("model_id"), row.get("requested_precision"))] = row
    rows = []
    for model_id, payload in ocr_metrics.items():
        for precision, metrics in (("fp32", payload.get("fp32")), ("INT8", payload.get("int8"))):
            if not metrics:
                continue
            timed = lat.get((model_id, precision), {})
            rows.append(
                {
                    "run_id": f"{model_id}_{precision}",
                    "model_id": model_id,
                    "requested_precision": precision,
                    "actual_precision_summary": timed.get("actual_precision_summary") or precision,
                    "accuracy": metrics.get("exact_normalized"),
                    "p50_ms": timed.get("p50_ms"),
                    "p95_ms": timed.get("p95_ms"),
                    "p99_ms": timed.get("p99_ms"),
                    "throughput": timed.get("throughput"),
                    "backend": metrics.get("backend") or timed.get("backend"),
                    "status": "measured",
                    "role": "TRAIN_HOST",
                    "hardware_fingerprint": hardware.get("nvidia_smi") or hardware.get("machine"),
                    "reason": f"exact_normalized on packaged {precision} ONNX",
                    "benchmark_on_target": False,
                    "cer": metrics.get("cer"),
                }
            )
    return rows
