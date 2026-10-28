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
from vn_alpr.data.convert import class_role
from vn_alpr.data.download import budget_ok, fetch_url, safe_extract_zip
from vn_alpr.data.prepare import prepare_locked_split
from vn_alpr.secrets import redact
from vn_alpr.textnorm import normalize_plate

TEXT_KEYS = ("plate_text", "text", "transcription")
SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}
CCT_ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CCT_MAX_LEN = 10


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
    # A YAML float can turn the class "0" into "0.0". That is still the digit class.
    if re.fullmatch(r"[0-9]\.0+", text):
        return text[0]
    # Some exports prefix the glyph, for example char_A or character-5.
    tail = text.replace("-", "_").split("_")[-1]
    if tail != text and len(tail) == 1 and tail.isalnum():
        return tail.upper()
    glyphs = [char for char in text if char.isalnum()]
    if len(glyphs) == 1 and any(not char.isalnum() for char in text):
        return glyphs[0].upper()
    return None


def _plate_string(name: str) -> str | None:
    """Use a class name as transcription only when it already looks like a plate string."""
    if class_role(str(name)) != "unknown":
        return None
    text = normalize_plate(name)
    if not (4 <= len(text) <= 12) or not text.isalnum():
        return None
    if not any(char.isdigit() for char in text) or not any(char.isalpha() for char in text):
        return None
    return text


def choose_listed_version(numbers: list[int], pinned: int | None) -> tuple[int | None, str]:
    """Pick a version the API listed. A missing pin is not replaced by a guessed number."""
    listed = sorted({int(number) for number in numbers})
    if pinned is not None:
        return int(pinned), "pinned"
    if not listed:
        return None, "none_listed"
    if len(listed) == 1:
        return listed[0], "only_listed"
    return listed[-1], "highest_listed"


def _id_keys(value: Any) -> list[Any]:
    """Integer 1 and string "1" are the same COCO id. Bools are not ids."""
    if isinstance(value, bool) or value is None:
        return []
    keys: list[Any] = [value]
    if isinstance(value, float) and value.is_integer():
        number = int(value)
        if number not in keys:
            keys.append(number)
        text = str(number)
    else:
        text = str(value).strip()
    if text and text not in keys:
        keys.append(text)
    if text.isdigit():
        number = int(text)
        if number not in keys:
            keys.append(number)
    elif re.fullmatch(r"\d+\.0+", text or ""):
        number = int(text.split(".", 1)[0])
        if number not in keys:
            keys.append(number)
        if str(number) not in keys:
            keys.append(str(number))
    return keys


def _append_under(buckets: dict[Any, list], keys: list[Any], item: Any) -> None:
    aliases: list[Any] = []
    for key in keys:
        for alias in _id_keys(key):
            if alias not in aliases:
                aliases.append(alias)
    if not aliases:
        return
    existing = None
    for alias in aliases:
        if alias in buckets:
            existing = buckets[alias]
            break
    if existing is None:
        buckets[aliases[0]].append(item)
        existing = buckets[aliases[0]]
    else:
        existing.append(item)
    for alias in aliases:
        buckets[alias] = existing


def _first_list(buckets: dict[Any, list], *values: Any) -> list:
    for value in values:
        for key in _id_keys(value):
            if key in buckets:
                return buckets[key]
    return []


def _put_text(texts: dict[Any, str], key: Any, value: str, *, overwrite: bool = False) -> None:
    aliases = _id_keys(key)
    if not aliases:
        return
    if not overwrite and any(alias in texts for alias in aliases):
        return
    for alias in aliases:
        texts[alias] = value


def _first_text(texts: dict[Any, str], *values: Any) -> str | None:
    for value in values:
        for key in _id_keys(value):
            if key in texts:
                return texts[key]
    return None


def _bbox_xywh(ann: dict[str, Any]) -> list[float] | None:
    bbox = ann.get("bbox")
    if isinstance(bbox, dict):
        x = bbox.get("x", bbox.get("left"))
        y = bbox.get("y", bbox.get("top"))
        width = bbox.get("width", bbox.get("w"))
        height = bbox.get("height", bbox.get("h"))
        try:
            if None not in (x, y, width, height):
                return [float(x), float(y), float(width), float(height)]
        except (TypeError, ValueError):
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except (TypeError, ValueError):
            pass
    segmentation = ann.get("segmentation")
    if not isinstance(segmentation, list):
        return None
    for part in segmentation:
        if not isinstance(part, (list, tuple)) or len(part) < 6:
            continue
        try:
            xs = [float(part[index]) for index in range(0, len(part), 2)]
            ys = [float(part[index]) for index in range(1, len(part), 2)]
        except (TypeError, ValueError):
            continue
        if xs and ys:
            return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return None


def _category_map(document: dict[str, Any], label_names: set[str] | None) -> dict[Any, str]:
    categories: dict[Any, str] = {}
    for item in document.get("categories") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        if name and label_names is not None:
            label_names.add(name)
        for key in _id_keys(item.get("id")):
            categories[key] = name
    return categories


def _annotation_name(ann: dict[str, Any], categories: dict[Any, str]) -> str:
    for key in _id_keys(ann.get("category_id")):
        if key in categories and categories[key]:
            return categories[key]
    attributes = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    for key in ("label", "class", "category_name", "class_name"):
        if ann.get(key):
            return str(ann[key])
        if attributes.get(key):
            return str(attributes[key])
    return ""


def _annotation_text(ann: dict[str, Any]) -> str | None:
    attributes = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    for key in TEXT_KEYS:
        if ann.get(key):
            return str(ann[key])
        if attributes.get(key):
            return str(attributes[key])
    return None


def _iter_annotations(document: dict[str, Any]):
    annotations = document.get("annotations") or []
    if annotations:
        for ann in annotations:
            if isinstance(ann, dict):
                yield ann
        return
    for image in document.get("images") or []:
        if not isinstance(image, dict):
            continue
        nested = image.get("annotations") or image.get("objects") or []
        for ann in nested:
            if not isinstance(ann, dict):
                continue
            copied = dict(ann)
            copied.setdefault("image_id", image.get("id"))
            yield copied


def samples_from_coco_tree(
    root: Path,
    dataset: dict[str, Any],
    label_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    samples = []
    seen: set[str] = set()
    for ann_path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(ann_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or "images" not in document:
            continue
        if "annotations" not in document and not any(
            isinstance(image, dict) and (image.get("annotations") or image.get("objects"))
            for image in document.get("images") or []
        ):
            continue
        categories = _category_map(document, label_names)
        boxes: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        chars: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        texts: dict[Any, str] = {}
        for ann in _iter_annotations(document):
            name = _annotation_name(ann, categories)
            if name and label_names is not None:
                label_names.add(name)
            bbox = _bbox_xywh(ann)
            link_keys = [ann.get("image_id"), ann.get("file_name")]
            _append_under(boxes, link_keys, {"name": name, "bbox": bbox})
            character = _character(name)
            if character and bbox:
                _append_under(
                    chars,
                    link_keys,
                    {"char": character, "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                )
            plate_string = _plate_string(name)
            if plate_string:
                _put_text(texts, ann.get("image_id"), plate_string)
                if ann.get("file_name"):
                    _put_text(texts, ann.get("file_name"), plate_string)
            explicit = _annotation_text(ann)
            if explicit:
                _put_text(texts, ann.get("image_id"), explicit, overwrite=True)
                if ann.get("file_name"):
                    _put_text(texts, ann.get("file_name"), explicit, overwrite=True)
        folder_name = ann_path.parent.name.lower()
        split_name = folder_name if folder_name in {"train", "valid", "val", "test"} else None
        for image in document.get("images", []):
            if not isinstance(image, dict):
                continue
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
            linked_boxes = _first_list(boxes, image.get("id"), file_name)
            linked_chars = _first_list(chars, image.get("id"), file_name)
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
                    "boxes": [box for box in linked_boxes if box.get("bbox")],
                    "char_boxes": linked_chars or None,
                    "text_raw": _first_text(texts, image.get("id"), file_name),
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


def _balance_ocr_splits(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Give both splits at least one existing label. Do not invent text."""
    if train_rows and val_rows:
        return train_rows, val_rows
    pool = list(train_rows) + list(val_rows)
    if len(pool) < 2:
        return train_rows, val_rows
    if not val_rows:
        return pool[:-1], pool[-1:]
    return pool[:1], pool[1:]


def _fits_cct(text: str) -> bool:
    return 1 <= len(text) <= CCT_MAX_LEN and all(char in CCT_ALPHABET for char in text)


def _ocr_sidecars(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = out_dir / "ocr.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = [row for row in rows if row.get("text_normalized") and row.get("split") in {"train", "val"} and row.get("image_path")]
    train_rows = [row for row in usable if row["split"] == "train"]
    val_rows = [row for row in usable if row["split"] == "val"]
    train_rows, val_rows = _balance_ocr_splits(train_rows, val_rows)
    if not train_rows or not val_rows:
        manifest["ocr_counts"] = {"train": len(train_rows), "val": len(val_rows), "cct": {}}
        return
    image_root = out_dir / "ocr_images"
    image_root.mkdir(parents=True, exist_ok=True)
    csv_paths = {}
    list_paths = {}
    cct_counts = {}
    for split, chosen in (("train", train_rows), ("val", val_rows)):
        csv_path = out_dir / f"{split}.csv"
        list_path = out_dir / f"{split}_list.txt"
        csv_lines = ["image_path,plate_text"]
        list_lines = []
        cct_rows = 0
        for index, row in enumerate(chosen):
            source = Path(row["image_path"])
            target = image_root / f"{split}_{index}{source.suffix.lower() or '.jpg'}"
            if not target.exists():
                os.symlink(source.resolve(), target)
            text = str(row["text_normalized"]).replace(",", "")
            list_lines.append(f"{target.name}\t{text}")
            if _fits_cct(text):
                # fast-plate-ocr joins the CSV directory with this field as a
                # string. An absolute path becomes canonical//canonical/...
                csv_lines.append(f"{target.relative_to(out_dir).as_posix()},{text}")
                cct_rows += 1
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
        if cct_rows:
            csv_paths[split] = str(csv_path)
        list_paths[split] = str(list_path)
        cct_counts[split] = cct_rows
    if csv_paths.get("train") and csv_paths.get("val"):
        manifest["ocr_csv"] = csv_paths
    manifest["paddle_lists"] = {
        "train_dir": str(image_root),
        "train_list": list_paths["train"],
        "val_list": list_paths["val"],
    }
    manifest["ocr_counts"] = {"train": len(train_rows), "val": len(val_rows), "cct": cct_counts}


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
        pinned, how = choose_listed_version(numbers, dataset.get("version"))
        if pinned is None:
            return {
                "dataset_id": dataset["dataset_id"],
                "status": "versions_listed",
                "versions": numbers,
                "downloaded": False,
            }
        dest.mkdir(parents=True, exist_ok=True)
        if _coco_cached(dest):
            return {
                "dataset_id": dataset["dataset_id"],
                "status": "downloaded",
                "version": "cached",
                "downloaded": True,
            }
        project.version(int(pinned)).download("coco", location=str(dest), overwrite=True)
    except Exception as exc:  # noqa: BLE001
        message = redact(str(exc))
        if api_key and api_key in message:
            message = message.replace(api_key, "[REDACTED]")
        return {"dataset_id": dataset["dataset_id"], "status": "failed_api", "reason": message, "downloaded": False}
    return {
        "dataset_id": dataset["dataset_id"],
        "status": "downloaded",
        "version": int(pinned),
        "version_choice": how,
        "downloaded": True,
    }


def _coco_cached(dest: Path) -> bool:
    if not dest.exists():
        return False
    for path in dest.rglob("*.json"):
        if not path.is_file():
            continue
        head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        if '"images"' in head and '"annotations"' in head:
            return True
    return False


def _download_rank(dataset: dict[str, Any]) -> tuple[int, str]:
    tasks = set(dataset.get("tasks_available") or [])
    # Character and OCR sets have to land before the download budget is spent on detection.
    if tasks & {"ocr", "character_detection"}:
        return 0, dataset["dataset_id"]
    return 1, dataset["dataset_id"]


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
    label_names: dict[str, list[str]] = {}
    max_gb = float(cfg["max_download_gb"])
    datasets = [dataset for dataset in load_datasets() if dataset["dataset_id"] in selected]
    datasets.sort(key=_download_rank)
    for dataset in datasets:
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
            seen_names: set[str] = set()
            samples.extend(samples_from_coco_tree(dest, dataset, seen_names))
            label_names[dataset["dataset_id"]] = sorted(seen_names, key=lambda name: (len(name), name))[:80]
    samples, scope = limit_samples(samples, cfg)
    if not samples:
        return {
            "status": "dataset_missing",
            "downloads": statuses,
            "counts": {},
            "score_scope": scope,
            "label_names": label_names,
            "ocr_text_samples": 0,
        }
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
        "ocr": bool((manifest.get("paddle_lists") or {}).get("train_list")),
        "ocr_counts": manifest.get("ocr_counts"),
        "ocr_text_samples": sum(1 for sample in samples if sample.get("char_boxes") or sample.get("text_raw")),
        "label_names": label_names,
        "manifest": str(out_dir / "split_manifest.json"),
    }
