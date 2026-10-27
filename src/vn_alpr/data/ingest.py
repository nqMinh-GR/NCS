"""Download public catalogs and lock a canonical split. No version is invented."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from vn_alpr.data import catalog_plan, load_datasets
from vn_alpr.data.download import budget_ok, fetch_url, safe_extract_zip
from vn_alpr.data.prepare import prepare_locked_split
from vn_alpr.secrets import redact

TEXT_KEYS = ("plate_text", "text", "transcription")
SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:90]


def _find_image(folder: Path, file_name: str) -> Path | None:
    direct = folder / file_name
    if direct.is_file():
        return direct
    name = Path(file_name).name
    matches = [path for path in folder.rglob(name) if path.is_file()]
    return matches[0] if len(matches) == 1 else None


def _character(name: str) -> str | None:
    text = str(name).strip()
    if len(text) == 1 and text.isalnum():
        return text.upper()
    return None


def samples_from_coco_tree(root: Path, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    seen: set[str] = set()
    for ann_path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(ann_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or "images" not in document or "annotations" not in document:
            continue
        categories = {item.get("id"): item.get("name", "") for item in document.get("categories", [])}
        boxes: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        chars: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        texts: dict[Any, str] = {}
        for ann in document.get("annotations", []):
            name = categories.get(ann.get("category_id"), "")
            bbox = ann.get("bbox")
            boxes[ann.get("image_id")].append({"name": name, "bbox": bbox})
            character = _character(name)
            if character and isinstance(bbox, list) and len(bbox) >= 4:
                chars[ann.get("image_id")].append(
                    {"char": character, "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
                )
            for key in TEXT_KEYS:
                if ann.get(key):
                    texts[ann.get("image_id")] = str(ann[key])
                    break
        folder_name = ann_path.parent.name.lower()
        split_name = folder_name if folder_name in {"train", "valid", "val", "test"} else None
        for image in document.get("images", []):
            file_name = image.get("file_name")
            if not file_name:
                continue
            path = _find_image(ann_path.parent, file_name)
            if path is None:
                continue
            sample_id = f"{dataset['dataset_id']}_{split_name or 'na'}_{image.get('id')}_{path.name}"
            if sample_id in seen:
                continue
            seen.add(sample_id)
            width = image.get("width")
            height = image.get("height")
            if not width or not height:
                with Image.open(path) as handle:
                    width, height = handle.size
            samples.append(
                {
                    "sample_id": sample_id,
                    "file_name": f"{_safe_stem(sample_id)}{path.suffix.lower() or '.jpg'}",
                    "image_path": str(path),
                    "width": int(width),
                    "height": int(height),
                    "dataset_id": dataset["dataset_id"],
                    "source_family_id": dataset.get("source_family_id") or dataset["dataset_id"],
                    "country": dataset.get("country"),
                    "role": dataset.get("role"),
                    "boxes": [box for box in boxes.get(image.get("id"), []) if box.get("bbox")],
                    "char_boxes": chars.get(image.get("id")) or None,
                    "text_raw": texts.get(image.get("id")),
                    "official_test": split_name == "test",
                }
            )
    return samples


def limit_samples(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    fraction = float(cfg.get("auxiliary", {}).get("max_fraction_of_train", 0.2))
    primary = [sample for sample in samples if sample.get("role") not in {"auxiliary", "mixed_auxiliary"}]
    auxiliary = [sample for sample in samples if sample.get("role") in {"auxiliary", "mixed_auxiliary"}]
    auxiliary.sort(key=lambda sample: sample["sample_id"])
    if primary:
        keep = int(len(primary) * fraction)
        auxiliary = auxiliary[:keep]
        scope = "includes_primary"
    else:
        scope = "mixed_auxiliary_not_vn_leaderboard" if auxiliary else "empty"
    chosen = primary + auxiliary
    chosen.sort(key=lambda sample: sample["sample_id"])
    max_images = cfg["training"].get("max_images")
    if max_images is not None:
        official = [sample for sample in chosen if sample.get("official_test")]
        rest = [sample for sample in chosen if not sample.get("official_test")]
        room = max(0, int(max_images) - len(official))
        chosen = official + rest[:room]
    return chosen, scope


def materialize_images(samples: list[dict[str, Any]], manifest: dict[str, Any], dest: Path) -> dict[str, str]:
    split_of: dict[str, str] = {}
    for group_id, members in manifest["groups"].items():
        split = manifest["assignment"][group_id]
        for sample_id in members:
            split_of[sample_id] = split
    image_dirs: dict[str, str] = {}
    for split, folder in SPLIT_FOLDER.items():
        path = dest / folder
        path.mkdir(parents=True, exist_ok=True)
        image_dirs[split] = str(path)
    by_id = {sample["sample_id"]: sample for sample in samples}
    for sample_id, split in split_of.items():
        sample = by_id.get(sample_id)
        if sample is None or split not in image_dirs:
            continue
        target = Path(image_dirs[split]) / sample["file_name"]
        source = Path(sample["image_path"]).resolve()
        if target.exists() or target.is_symlink():
            continue
        os.symlink(source, target)
    return image_dirs


def _write_annotations(out_dir: Path, image_dirs: dict[str, str]) -> None:
    for split, folder in SPLIT_FOLDER.items():
        source = out_dir / f"instances_{split}.json"
        if source.exists():
            (Path(image_dirs[split]) / "_annotations.coco.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _ocr_sidecars(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = out_dir / "ocr.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = [row for row in rows if row.get("text_normalized") and row.get("split") in {"train", "val"} and row.get("image_path")]
    train_rows = [row for row in usable if row["split"] == "train"]
    val_rows = [row for row in usable if row["split"] == "val"]
    if not train_rows or not val_rows:
        return
    image_root = out_dir / "ocr_images"
    image_root.mkdir(parents=True, exist_ok=True)
    csv_paths = {}
    list_paths = {}
    for split, chosen in (("train", train_rows), ("val", val_rows)):
        csv_path = out_dir / f"{split}.csv"
        list_path = out_dir / f"{split}_list.txt"
        csv_lines = ["image_path,plate_text"]
        list_lines = []
        for index, row in enumerate(chosen):
            source = Path(row["image_path"])
            target = image_root / f"{split}_{index}{source.suffix.lower() or '.jpg'}"
            if not target.exists():
                os.symlink(source.resolve(), target)
            text = str(row["text_normalized"]).replace(",", "")
            csv_lines.append(f"{target},{text}")
            list_lines.append(f"{target.name}\t{text}")
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
        csv_paths[split] = str(csv_path)
        list_paths[split] = str(list_path)
    manifest["ocr_csv"] = csv_paths
    manifest["paddle_lists"] = {
        "train_dir": str(image_root),
        "train_list": list_paths["train"],
        "val_list": list_paths["val"],
    }


def download_hf_zips(dataset: dict[str, Any], dest: Path, max_gb: float) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in dataset.get("files") or []:
        ok, reason = budget_ok(dest, max_gb, 2.0)
        if not ok:
            return {"dataset_id": dataset["dataset_id"], "status": "budget", "reason": reason}
        target = dest / Path(name).name
        url = (
            f"https://huggingface.co/datasets/{dataset['hf_id']}/resolve/"
            f"{dataset['hf_revision']}/{name}?download=true"
        )
        try:
            fetch_url(url, target, timeout=180)
            safe_extract_zip(target, dest / "extracted" / target.stem)
        except Exception as exc:  # noqa: BLE001
            return {"dataset_id": dataset["dataset_id"], "status": "failed", "reason": str(exc)}
        saved.append(target.name)
    return {"dataset_id": dataset["dataset_id"], "status": "downloaded", "files": saved}


def download_roboflow_coco(dataset: dict[str, Any], api_key: str, dest: Path) -> dict[str, Any]:
    if dataset.get("version") is None and dataset.get("access_status") == "no_export_version_observed":
        return {"dataset_id": dataset["dataset_id"], "status": "version_unavailable", "downloaded": False}
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        return {"dataset_id": dataset["dataset_id"], "status": "dependency_missing", "reason": "pip install roboflow"}
    try:
        project = Roboflow(api_key=api_key).workspace(dataset["workspace"]).project(dataset["project"])
        numbers = []
        for version in project.versions():
            number = getattr(version, "version", None) or getattr(version, "version_number", None)
            if number is not None:
                numbers.append(int(number))
        pinned = dataset.get("version")
        if pinned is None:
            if len(numbers) != 1:
                return {
                    "dataset_id": dataset["dataset_id"],
                    "status": "versions_listed",
                    "versions": numbers,
                    "downloaded": False,
                }
            pinned = numbers[0]
        dest.mkdir(parents=True, exist_ok=True)
        project.version(int(pinned)).download("coco", location=str(dest), overwrite=True)
    except Exception as exc:  # noqa: BLE001
        message = redact(str(exc))
        if api_key and api_key in message:
            message = message.replace(api_key, "[REDACTED]")
        return {"dataset_id": dataset["dataset_id"], "status": "failed_api", "reason": message, "downloaded": False}
    return {"dataset_id": dataset["dataset_id"], "status": "downloaded", "version": int(pinned), "downloaded": True}


def prepare_training_data(
    cfg: dict[str, Any],
    raw_dir: Path,
    out_dir: Path,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Download eligible catalogs, cap auxiliary data, and write the locked split."""
    selected = {row["dataset_id"] for row in catalog_plan(cfg["commercial_only"]) if row["selected_for_default_download"]}
    statuses = []
    samples: list[dict[str, Any]] = []
    max_gb = float(cfg["max_download_gb"])
    for dataset in load_datasets():
        if dataset["dataset_id"] not in selected:
            continue
        dest = raw_dir / dataset["dataset_id"]
        method = dataset.get("download_method")
        status: dict[str, Any]
        if method == "huggingface_resolve_zip":
            status = download_hf_zips(dataset, dest, max_gb)
        elif method == "roboflow_sdk_coco":
            if not api_key:
                status = {"dataset_id": dataset["dataset_id"], "status": "credentials_required", "downloaded": False}
            else:
                status = download_roboflow_coco(dataset, api_key, dest)
        else:
            continue
        statuses.append(status)
        if status.get("status") == "downloaded":
            samples.extend(samples_from_coco_tree(dest, dataset))
    samples, scope = limit_samples(samples, cfg)
    if not samples:
        return {"status": "dataset_missing", "downloads": statuses, "counts": {}, "score_scope": scope}
    out_dir.mkdir(parents=True, exist_ok=True)
    official = [sample["sample_id"] for sample in samples if sample.get("official_test")]
    manifest = prepare_locked_split(
        samples,
        out_dir,
        seed=int(cfg["seed"]),
        official_test_ids=official,
        calibration_n=int(cfg["quantization"]["calibration_samples"]),
        self_test=False,
    )
    image_dirs = materialize_images(samples, manifest, out_dir / "rfdetr_data")
    _write_annotations(out_dir, image_dirs)
    manifest["image_dirs"] = image_dirs
    manifest["rfdetr_dir"] = str(out_dir / "rfdetr_data")
    manifest["score_scope"] = scope
    _ocr_sidecars(out_dir, manifest)
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "status": "prepared",
        "downloads": statuses,
        "counts": manifest["counts"],
        "score_scope": scope,
        "ocr": bool(manifest.get("ocr_csv")),
        "manifest": str(out_dir / "split_manifest.json"),
    }
