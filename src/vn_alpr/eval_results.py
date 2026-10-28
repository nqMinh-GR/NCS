"""Run OCR eval, INT8 PTQ, and Colab GPU latency on a packaged vn_alpr_results tree."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from vn_alpr.benchmark.host import _open_session, measure_onnx_files
from vn_alpr.evaluation.combined import evaluate_chain
from vn_alpr.evaluation.detection import evaluate_detection
from vn_alpr.evaluation.ocr import evaluate_ocr
from vn_alpr.geometry import letterbox_xywh_to_original, xyxy_to_xywh
from vn_alpr.quantization.precision import quality_gate
from vn_alpr.quantization.ptq_onnx import quantize_onnx_int8
from vn_alpr.reporting import write_summary, write_table

DET_IMAGE_FOLDERS = {"train": "train", "val": "valid", "test": "test"}

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


def load_ocr_text_by_sample(canonical: Path) -> dict[str, str]:
    path = canonical / "ocr.jsonl"
    if not path.is_file():
        return {}
    found: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("sample_id") or "")
        text = str(row.get("text_normalized") or row.get("text_raw") or "").strip()
        if not sample_id or not text:
            continue
        found.setdefault(sample_id, set()).add(text)
    return {key: next(iter(values)) for key, values in found.items() if len(values) == 1}


def resolve_detection_image(canonical: Path, file_name: str, split: str) -> Path | None:
    name = Path(file_name).name
    folder = DET_IMAGE_FOLDERS.get(split, split)
    candidates = [
        canonical / "rfdetr_data" / folder / name,
        canonical / "rfdetr_data" / folder / file_name,
        canonical / "rfdetr_data" / split / name,
        canonical / folder / name,
        canonical / split / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_detection_split(
    canonical: Path,
    split: str,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    coco_path = canonical / f"instances_{split}.json"
    if not coco_path.is_file():
        return []
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    texts = load_ocr_text_by_sample(canonical)
    anns: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations") or []:
        anns.setdefault(int(ann["image_id"]), []).append(ann)
    rows = []
    missing = 0
    for image in coco.get("images") or []:
        path = resolve_detection_image(canonical, str(image.get("file_name") or ""), split)
        if path is None:
            missing += 1
            continue
        sample_id = str(image.get("sample_id") or "")
        text = texts.get(sample_id)
        gt = []
        boxes = anns.get(int(image["id"]), [])
        for ann in boxes:
            item = {"bbox": [float(v) for v in ann["bbox"]]}
            if text and len(boxes) == 1:
                item["text"] = text
            gt.append(item)
        rows.append(
            {
                "image_path": path,
                "sample_id": sample_id,
                "width": int(image.get("width") or 0),
                "height": int(image.get("height") or 0),
                "gt": gt,
                "source": image.get("source") or "results_zip",
            }
        )
        if max_images is not None and len(rows) >= int(max_images):
            break
    if missing:
        print(f"detection images missing {missing} on {split}", flush=True)
    return rows


def softmax_last(values):
    import numpy as np

    arr = np.asarray(values, dtype="float32")
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-9, None)


def decode_rfdetr_boxes(dets, labels, orig_w: int, orig_h: int, params: dict[str, float], score_threshold: float) -> list[dict[str, Any]]:
    import numpy as np

    boxes = np.asarray(dets)
    logits = np.asarray(labels)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if logits.ndim == 3:
        logits = logits[0]
    scores = softmax_last(logits)[:, 0]
    size = float(params["dst"])
    out = []
    for box, score in zip(boxes, scores):
        if float(score) < score_threshold:
            continue
        cx, cy, bw, bh = (float(v) for v in box)
        xywh_canvas = ((cx - bw / 2.0) * size, (cy - bh / 2.0) * size, bw * size, bh * size)
        x, y, w, h = letterbox_xywh_to_original(xywh_canvas, params)
        out.append({"bbox": [x, y, w, h], "score": float(score), "text": ""})
    return out


def decode_xyxy_boxes(boxes, scores, score_threshold: float) -> list[dict[str, Any]]:
    import numpy as np

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    out = []
    for box, score in zip(boxes, scores):
        if float(score) < score_threshold:
            continue
        x, y, w, h = xyxy_to_xywh(box)
        out.append({"bbox": [x, y, w, h], "score": float(score), "text": ""})
    return out


def decode_detector_outputs(named: dict[str, Any], meta: dict[str, Any], score_threshold: float) -> list[dict[str, Any]]:
    if "dets" in named and "labels" in named:
        return decode_rfdetr_boxes(named["dets"], named["labels"], meta["orig_w"], meta["orig_h"], meta["params"], score_threshold)
    if "boxes" in named and "scores" in named:
        return decode_xyxy_boxes(named["boxes"], named["scores"], score_threshold)
    raise ValueError(f"unsupported detector outputs: {sorted(named)}")


def crop_xywh(image, box, pad: float = 0.1):
    x, y, w, h = (float(v) for v in box)
    px, py = w * pad, h * pad
    x1 = max(0, int(x - px))
    y1 = max(0, int(y - py))
    x2 = min(int(image.shape[1]), int(x + w + px))
    y2 = min(int(image.shape[0]), int(y + h + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


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


def preprocess_detector(image, session, model_id: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image
    import numpy as np

    feeds = {}
    height, width = int(image.shape[0]), int(image.shape[1])
    rfdetr = "rfdetr" in (model_id or str(getattr(session, "_model_path", ""))).lower()
    params = None
    size = 640
    for item in session.get_inputs():
        raw = list(item.shape)
        for dim in raw[1:]:
            if isinstance(dim, int) and dim >= 32:
                size = int(dim)
                break
    rgb = Image.fromarray(image)
    if rfdetr:
        scale = min(size / width, size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = rgb.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) / 2.0
        pad_y = (size - new_h) / 2.0
        canvas.paste(resized, (int(pad_x), int(pad_y)))
        arr = np.asarray(canvas).astype("float32") / 255.0
        nchw = np.transpose(arr, (2, 0, 1))[None]
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype="float32")[:, None, None]
        nchw = (nchw - mean) / std
        params = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "dst": float(size)}
    else:
        arr = np.asarray(rgb.resize((size, size), Image.BILINEAR)).astype("float32") / 255.0
        nchw = np.transpose(arr, (2, 0, 1))[None]
    for item in session.get_inputs():
        raw = list(item.shape)
        kind = item.type or ""
        if "int64" in kind:
            feeds[item.name] = np.array([[width, height]], dtype=np.int64)
            continue
        if raw and isinstance(raw[0], int) and raw[0] > 0 and nchw.shape[0] != raw[0]:
            raise ValueError(f"fixed batch {raw[0]} cannot take {nchw.shape[0]}")
        feeds[item.name] = nchw
    meta = {"orig_w": width, "orig_h": height, "params": params, "size": size, "model_id": model_id, "rfdetr": rfdetr}
    return feeds, meta


def _session(path: Path, prefer_cuda: bool):
    session, provider = _open_session(path, prefer_cuda=prefer_cuda)
    session._model_path = str(path)
    return session, provider


def read_ocr_text(session, image, plate_cfg: dict[str, Any]) -> str:
    alphabet = str(plate_cfg.get("alphabet") or DEFAULT_ALPHABET)
    pad = str(plate_cfg.get("pad_char") or DEFAULT_PAD)
    outputs = session.run(None, preprocess_ocr(image, session, plate_cfg))
    return decode_slots(outputs[0], alphabet, pad)


def eval_ocr_model(
    path: Path,
    rows: list[dict[str, Any]],
    plate_cfg: dict[str, Any],
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    session, provider = _session(path, prefer_cuda)
    samples = []
    times = []
    for row in rows:
        image = _read_rgb(row["image_path"])
        tick = __import__("time").perf_counter()
        pred = read_ocr_text(session, image, plate_cfg)
        times.append((__import__("time").perf_counter() - tick) * 1000.0)
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
    if times:
        metrics["real_image_p50_ms"] = sorted(times)[len(times) // 2]
        metrics["real_image_mean_ms"] = sum(times) / len(times)
    return metrics


def run_detector(session, image, model_id: str, score_threshold: float) -> list[dict[str, Any]]:
    feeds, meta = preprocess_detector(image, session, model_id)
    named = dict(zip([item.name for item in session.get_outputs()], session.run(None, feeds)))
    return decode_detector_outputs(named, meta, score_threshold)


def eval_detection_model(
    path: Path,
    rows: list[dict[str, Any]],
    model_id: str,
    score_threshold: float,
    iou_match: float,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    session, provider = _session(path, prefer_cuda)
    images = []
    times = []
    for index, row in enumerate(rows):
        image = _read_rgb(row["image_path"])
        tick = __import__("time").perf_counter()
        preds = run_detector(session, image, model_id, score_threshold)
        times.append((__import__("time").perf_counter() - tick) * 1000.0)
        images.append({"gt": row["gt"], "preds": preds, "source": row.get("source")})
        if index and index % 50 == 0:
            print(f"  {model_id} detection {index}/{len(rows)}", flush=True)
    metrics = evaluate_detection(images, score_threshold=score_threshold, iou_match=iou_match)
    metrics.update({"backend": provider, "n_images": len(images), "model_path": str(path), "images": images})
    if times:
        metrics["real_image_p50_ms"] = sorted(times)[len(times) // 2]
        metrics["real_image_mean_ms"] = sum(times) / len(times)
    return metrics


def eval_pipeline(
    det_path: Path,
    ocr_path: Path,
    rows: list[dict[str, Any]],
    plate_cfg: dict[str, Any],
    det_id: str,
    score_threshold: float,
    iou_match: float,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    det_session, det_provider = _session(det_path, prefer_cuda)
    ocr_session, ocr_provider = _session(ocr_path, prefer_cuda)
    images = []
    times = []
    oracle = []
    for index, row in enumerate(rows):
        image = _read_rgb(row["image_path"])
        tick = __import__("time").perf_counter()
        preds = run_detector(det_session, image, det_id, score_threshold)
        for pred in preds:
            crop = crop_xywh(image, pred["bbox"])
            pred["text"] = read_ocr_text(ocr_session, crop, plate_cfg) if crop is not None else ""
        times.append((__import__("time").perf_counter() - tick) * 1000.0)
        images.append({"gt": row["gt"], "preds": preds, "source": row.get("source")})
        for box in row["gt"]:
            if not box.get("text"):
                continue
            crop = crop_xywh(image, box["bbox"])
            pred_text = read_ocr_text(ocr_session, crop, plate_cfg) if crop is not None else ""
            oracle.append({"gt_raw": box["text"], "pred_raw": pred_text, "readable": True, "source": "gt_box"})
        if index and index % 50 == 0:
            print(f"  pipeline {det_id} {index}/{len(rows)}", flush=True)
    labeled = [{"gt": [g for g in item["gt"] if g.get("text")], "preds": item["preds"]} for item in images]
    labeled = [item for item in labeled if item["gt"]]
    chain = evaluate_chain(labeled, score_threshold=score_threshold, iou_match=iou_match)
    oracle_metrics = evaluate_ocr(oracle) if oracle else {"n": 0, "exact_normalized": None}
    chain.update(
        {
            "backend_det": det_provider,
            "backend_ocr": ocr_provider,
            "images_with_text": len(labeled),
            "ocr_on_gt_box": oracle_metrics,
            "n_images": len(images),
        }
    )
    if times:
        chain["e2e_p50_ms"] = sorted(times)[len(times) // 2]
        chain["e2e_mean_ms"] = sum(times) / len(times)
        chain["e2e_p95_ms"] = sorted(times)[int(0.95 * (len(times) - 1))]
    return chain


def _calibration_feeds(
    path: Path,
    rows: list[dict[str, Any]],
    plate_cfg: dict[str, Any],
    kind: str,
    limit: int,
    model_id: str = "",
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
                item, _meta = preprocess_detector(image, session, model_id)
                feeds.append(item)
            except Exception:
                continue
    return feeds


def _strip_images(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "images"}


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
    score_thr = float((cfg.get("detection") or {}).get("score_threshold") or 0.5)
    iou_match = float((cfg.get("detection") or {}).get("iou_match") or 0.5)
    ocr_rows = load_ocr_rows(canonical, split, max_images=max_images)
    det_rows = load_detection_split(canonical, split, max_images=max_images)
    calib_n = int((cfg.get("quantization") or {}).get("calibration_samples") or 256)
    ocr_calib = load_ocr_rows(canonical, "train", max_images=calib_n) or ocr_rows[:calib_n]
    det_calib = load_detection_split(canonical, "train", max_images=calib_n) or det_rows[:calib_n]
    quant_dir = out_dir / "int8"
    quant_dir.mkdir(parents=True, exist_ok=True)

    ocr_metrics: dict[str, Any] = {}
    det_metrics: dict[str, Any] = {}
    chain_metrics: dict[str, Any] = {}
    quant_rows: list[dict[str, Any]] = []
    int8_by_id: dict[str, Path] = {}
    fp32_files: list[tuple[str, Path]] = []
    ocr_models = [m for m in models if m["kind"] == "ocr"]
    det_models = [m for m in models if m["kind"] == "detector"]
    primary_ocr = ocr_models[0] if ocr_models else None

    for model in models:
        model_id = model["model_id"]
        path = model["path"]
        fp32_files.append((model_id, path))
        print(f"--- quantize {model_id} ---", flush=True)
        calib = ocr_calib if model["kind"] == "ocr" else det_calib
        feeds = _calibration_feeds(path, calib, plate_cfg, model["kind"], calib_n, model_id)
        dest = quant_dir / f"{model_id}.onnx"
        quant = quantize_onnx_int8(path, dest, feeds)
        quant["model_id"] = model_id
        quant["kind"] = model["kind"]
        quant_rows.append(quant)
        print(model_id, quant.get("status"), quant.get("method"), quant.get("reason", "")[:180], flush=True)
        if quant.get("status") == "ok" and dest.is_file():
            int8_by_id[model_id] = dest

    for model in ocr_models:
        model_id = model["model_id"]
        print(f"--- ocr {model_id} {split} n={len(ocr_rows)} ---", flush=True)
        fp32 = eval_ocr_model(model["path"], ocr_rows, plate_cfg, prefer_cuda=prefer_cuda)
        payload = {"fp32": fp32}
        dest = int8_by_id.get(model_id)
        if dest is not None:
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
        print(model_id, "fp32_exact", fp32.get("exact_normalized"), "int8_exact", (payload.get("int8") or {}).get("exact_normalized"), flush=True)

    for model in det_models:
        model_id = model["model_id"]
        print(f"--- detect {model_id} {split} n={len(det_rows)} ---", flush=True)
        fp32 = eval_detection_model(model["path"], det_rows, model_id, score_thr, iou_match, prefer_cuda=prefer_cuda)
        payload = {"fp32": _strip_images(fp32)}
        dest = int8_by_id.get(model_id)
        if dest is not None:
            try:
                int8 = eval_detection_model(dest, det_rows, model_id, score_thr, iou_match, prefer_cuda=prefer_cuda)
                payload["int8"] = _strip_images(int8)
                payload["gate"] = quality_gate(
                    float(fp32["ap50"]),
                    float(int8["ap50"]),
                    float((cfg.get("quantization") or {}).get("max_detector_recall_drop_pp") or 0.5),
                )
            except Exception as exc:
                payload["int8_error"] = str(exc)[-400:]
        det_metrics[model_id] = payload
        print(model_id, "fp32_ap50", fp32.get("ap50"), "recall", fp32.get("recall_at_threshold"), "p50_ms", fp32.get("real_image_p50_ms"), flush=True)

    if primary_ocr and det_models and det_rows:
        for model in det_models:
            for precision, det_path, ocr_path in (
                ("fp32", model["path"], primary_ocr["path"]),
                ("INT8", int8_by_id.get(model["model_id"]), int8_by_id.get(primary_ocr["model_id"])),
            ):
                if det_path is None or ocr_path is None:
                    continue
                key = f"{model['model_id']}+{primary_ocr['model_id']}_{precision}"
                print(f"--- pipeline {key} n={len(det_rows)} ---", flush=True)
                try:
                    chain = eval_pipeline(
                        det_path,
                        ocr_path,
                        det_rows,
                        plate_cfg,
                        model["model_id"],
                        score_thr,
                        iou_match,
                        prefer_cuda=prefer_cuda,
                    )
                except Exception as exc:
                    chain = {"status": "failed", "reason": str(exc)[-400:]}
                chain_metrics[key] = chain
                print(key, "chain_exact", chain.get("chain_exact"), "e2e_p50_ms", chain.get("e2e_p50_ms"), flush=True)

    print("--- synthetic latency fp32 ---", flush=True)
    fp32_latency = measure_onnx_files(fp32_files, cfg, hardware, requested_precision="fp32")
    print("--- synthetic latency int8 ---", flush=True)
    int8_files = [(model_id, path) for model_id, path in int8_by_id.items()]
    int8_latency = measure_onnx_files(int8_files, cfg, hardware, requested_precision="INT8") if int8_files else []

    report = {
        "status": "implemented",
        "split": split,
        "results_dir": str(results_dir),
        "ocr_images": len(ocr_rows),
        "detection_images": len(det_rows),
        "models": [{"model_id": m["model_id"], "kind": m["kind"], "path": str(m["path"])} for m in models],
        "quantization": quant_rows,
        "ocr": ocr_metrics,
        "detection": det_metrics,
        "pipeline": chain_metrics,
        "fp32_latency": fp32_latency,
        "int8_latency": int8_latency,
        "benchmark_on_target": False,
        "role": "TRAIN_HOST",
        "primary_ocr": None if primary_ocr is None else primary_ocr["model_id"],
    }
    (out_dir / "int8_eval.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_table(out_dir / "int8_quantization.csv", [_quant_row(item, hardware) for item in quant_rows])
    write_table(out_dir / "fp32_latency.csv", fp32_latency)
    write_table(out_dir / "int8_latency.csv", int8_latency)
    summary_rows = _summary_rows(ocr_metrics, det_metrics, chain_metrics, fp32_latency, int8_latency, hardware)
    write_table(out_dir / "ocr_int8_leaderboard.csv", summary_rows)
    write_table(out_dir / "detector_int8_leaderboard.csv", [row for row in summary_rows if row.get("protocol") == "detection"])
    write_table(out_dir / "pipeline_int8_leaderboard.csv", [row for row in summary_rows if row.get("protocol") == "pipeline"])
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


def _base_row(model_id: str, precision: str, hardware: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": f"{model_id}_{precision}",
        "model_id": model_id,
        "requested_precision": precision,
        "status": "measured",
        "role": "TRAIN_HOST",
        "hardware_fingerprint": hardware.get("nvidia_smi") or hardware.get("machine"),
        "benchmark_on_target": False,
    }


def _summary_rows(
    ocr_metrics: dict[str, Any],
    det_metrics: dict[str, Any],
    chain_metrics: dict[str, Any],
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
    for model_id, payload in det_metrics.items():
        for precision, metrics in (("fp32", payload.get("fp32")), ("INT8", payload.get("int8"))):
            if not metrics:
                continue
            timed = lat.get((model_id, precision), {})
            row = _base_row(model_id, precision, hardware)
            row.update(
                {
                    "protocol": "detection",
                    "accuracy": metrics.get("ap50"),
                    "p50_ms": metrics.get("real_image_p50_ms") or timed.get("p50_ms"),
                    "p95_ms": timed.get("p95_ms"),
                    "throughput": timed.get("throughput"),
                    "backend": metrics.get("backend") or timed.get("backend"),
                    "reason": f"AP50={metrics.get('ap50')} recall={metrics.get('recall_at_threshold')} precision={metrics.get('precision_at_threshold')}",
                }
            )
            rows.append(row)
    for model_id, payload in ocr_metrics.items():
        for precision, metrics in (("fp32", payload.get("fp32")), ("INT8", payload.get("int8"))):
            if not metrics:
                continue
            timed = lat.get((model_id, precision), {})
            row = _base_row(model_id, precision, hardware)
            row.update(
                {
                    "protocol": "ocr",
                    "accuracy": metrics.get("exact_normalized"),
                    "p50_ms": metrics.get("real_image_p50_ms") or timed.get("p50_ms"),
                    "p95_ms": timed.get("p95_ms"),
                    "throughput": timed.get("throughput"),
                    "backend": metrics.get("backend") or timed.get("backend"),
                    "reason": f"exact_normalized={metrics.get('exact_normalized')} cer={metrics.get('cer')}",
                    "cer": metrics.get("cer"),
                }
            )
            rows.append(row)
    for key, chain in chain_metrics.items():
        if chain.get("chain_exact") is None:
            continue
        precision = "INT8" if key.endswith("_INT8") else "fp32"
        row = _base_row(key, precision, hardware)
        row.update(
            {
                "protocol": "pipeline",
                "accuracy": chain.get("chain_exact"),
                "p50_ms": chain.get("e2e_p50_ms"),
                "p95_ms": chain.get("e2e_p95_ms"),
                "backend": f"{chain.get('backend_det')}+{chain.get('backend_ocr')}",
                "reason": (
                    f"chain_exact={chain.get('chain_exact')} "
                    f"ocr_given_det={chain.get('ocr_exact_given_detection')} "
                    f"gt_box_ocr={((chain.get('ocr_on_gt_box') or {}).get('exact_normalized'))}"
                ),
            }
        )
        rows.append(row)
    return rows
