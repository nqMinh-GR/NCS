"""Zip run artifacts without raw datasets, teacher weights, or secrets."""

from __future__ import annotations

import zipfile
from pathlib import Path

from vn_alpr.dedup import sha256_file
from vn_alpr.secrets import find_secrets, path_is_sensitive

SKIP_DIR_NAMES = {
    "raw",
    "cache",
    "credentials",
    "teacher",
    ".git",
    ".github",
    "__pycache__",
    "venvs",
    ".venv",
    "upstream",
}
SKIP_SUFFIXES = {".pth", ".pdparams", ".engine"}


def should_pack(path: Path, root: Path) -> bool:
    if path_is_sensitive(path):
        return False
    relative = path.relative_to(root)
    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES and "onnx" not in path.name:
        return False
    if path.stat().st_size > 200 * 1024 * 1024:
        return False
    return True


def build_results_zip(run_dir: Path, zip_path: Path) -> dict[str, str | int]:
    files = []
    if run_dir.exists():
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and should_pack(path, run_dir):
                files.append(path)
    problems = []
    for path in files:
        if path.suffix.lower() not in {".json", ".yaml", ".yml", ".md", ".txt", ".log", ".csv", ".py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if find_secrets(text):
            problems.append(str(path))
    if problems:
        raise RuntimeError(f"refusing to zip secrets: {problems}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if run_dir.exists():
            for path in sorted(run_dir.rglob("*")):
                if path.is_file() and should_pack(path, run_dir):
                    archive.write(path, arcname=str(path.relative_to(run_dir)))
                    count += 1
    return {"zip": str(zip_path), "files": count, "sha256": sha256_file(zip_path), "bytes": zip_path.stat().st_size}
