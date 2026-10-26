"""Downloads that fail closed on HTML, zip-slip, missing credentials, and budget."""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from vn_alpr.secrets import redact

HTML_MARKERS = (b"<!doctype html", b"<html", b"<head")


class DownloadError(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return usage.free / (1024 ** 3)


def looks_like_html(header: bytes) -> bool:
    sniff = header.lstrip()[:64].lower()
    return any(sniff.startswith(marker) for marker in HTML_MARKERS)


def assert_archive(path: Path) -> str:
    data = path.read_bytes()[:8]
    if looks_like_html(data):
        raise DownloadError("download returned HTML, not an archive")
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
        return "zip"
    if data.startswith(b"\x1f\x8b"):
        return "gzip"
    raise DownloadError(f"unrecognized archive magic: {data[:4]!r}")


def safe_extract_zip(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise DownloadError(f"corrupt zip member {bad}")
        for info in archive.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise DownloadError(f"zip slip blocked: {name}")
            target = (root / name).resolve()
            if target != root and root not in target.parents:
                raise DownloadError(f"zip slip blocked: {name}")
            count += 0 if info.is_dir() else 1
        archive.extractall(root)
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "vn-alpr-lab"})
    with urlopen(request, timeout=timeout) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def gdrive_file_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_gdrive_binary(file_id: str, dest: Path) -> dict[str, Any]:
    """Save a Drive file that is not an archive. HTML interstitial pages are failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        fetch_url(url, dest, timeout=120)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return {"status": "failed", "reason": str(exc), "file_id": file_id}
    header = dest.read_bytes()[:64]
    if looks_like_html(header):
        dest.unlink(missing_ok=True)
        return {"status": "failed", "reason": "Google Drive returned HTML instead of the checkpoint", "file_id": file_id}
    return {"status": "downloaded", "path": str(dest), "sha256": sha256_file(dest)}


def download_gdrive_file(file_id: str, dest: Path) -> dict[str, Any]:
    """Download one Drive file. HTML interstitials are errors, not archives."""
    fetch_url(gdrive_file_url(file_id), dest)
    try:
        kind = assert_archive(dest)
    except DownloadError as exc:
        dest.unlink(missing_ok=True)
        return {"status": "failed", "reason": str(exc), "file_id": file_id}
    return {"status": "downloaded", "kind": kind, "path": str(dest), "sha256": sha256_file(dest)}


def budget_ok(dest: Path, max_gb: float, incoming_gb: float = 0.0) -> tuple[bool, str]:
    free = free_gb(dest)
    if incoming_gb > max_gb:
        return False, f"incoming {incoming_gb:.2f} GB exceeds max_download_gb {max_gb}"
    if free < incoming_gb + 1:
        return False, f"free space {free:.2f} GB is below the requested download"
    return True, "ok"


def roboflow_status(dataset: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    if dataset.get("access_status") == "no_export_version_observed" and dataset.get("version") is None:
        return {
            "dataset_id": dataset["dataset_id"],
            "status": "version_unavailable",
            "reason": dataset.get("reason"),
        }
    if not api_key:
        return {
            "dataset_id": dataset["dataset_id"],
            "status": "credentials_required",
            "reason": "set ROBOFLOW_API_KEY or provide a manual zip; the key is not printed",
        }
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        return {
            "dataset_id": dataset["dataset_id"],
            "status": "dependency_missing",
            "reason": "pip install roboflow",
        }
    try:
        project = Roboflow(api_key=api_key).workspace(dataset["workspace"]).project(dataset["project"])
        versions = project.versions()
        numbers = []
        for version in versions:
            number = getattr(version, "version", None) or getattr(version, "version_number", None)
            numbers.append(number)
        return {
            "dataset_id": dataset["dataset_id"],
            "status": "versions_listed",
            "versions": numbers,
            "pinned": dataset.get("version"),
        }
    except Exception as exc:  # noqa: BLE001
        message = redact(str(exc))
        if api_key and api_key in message:
            message = message.replace(api_key, "[REDACTED]")
        return {"dataset_id": dataset["dataset_id"], "status": "failed_api", "reason": message}


def manual_zip(dataset_id: str, path: Path, dest: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dataset_id": dataset_id, "status": "manual_zip_missing", "path": str(path)}
    try:
        assert_archive(path)
        count = safe_extract_zip(path, dest)
    except (DownloadError, zipfile.BadZipFile) as exc:
        return {"dataset_id": dataset_id, "status": "failed", "reason": str(exc)}
    return {
        "dataset_id": dataset_id,
        "status": "imported_manual",
        "files": count,
        "sha256": sha256_file(path),
        "permission": "manual_import_is_not_a_license_grant",
    }


def env_api_key() -> str | None:
    value = os.environ.get("ROBOFLOW_API_KEY")
    if value:
        return value
    return None
