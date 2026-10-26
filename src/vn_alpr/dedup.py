"""Identity hashes. Near-duplicates are audited, not silently merged."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image(path: Path | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        image = path
    else:
        image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def pixel_sha256(path: Path | Image.Image) -> str:
    image = canonical_image(path)
    payload = f"{image.size[0]}x{image.size[1]}".encode() + image.tobytes()
    return sha256_bytes(payload)


def average_hash(path: Path | Image.Image, size: int = 8) -> int:
    image = canonical_image(path).convert("L").resize((size, size), Image.Resampling.BOX)
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= mean)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_records(records: list[dict[str, Any]], phash_audit_max: int = 4) -> dict[str, Any]:
    """Union exact file or pixel identity, video id, and explicit plate identity.

    Source family alone does not merge a dataset into one group.
    Average-hash distance in 0..phash_audit_max is an audit candidate only.
    """
    uf = UnionFind()
    by_file: dict[str, list[str]] = {}
    by_pixel: dict[str, list[str]] = {}
    by_video: dict[str, list[str]] = {}
    by_plate: dict[str, list[str]] = {}
    for record in records:
        rid = str(record["sample_id"])
        uf.add(rid)
        by_file.setdefault(record["file_sha256"], []).append(rid)
        by_pixel.setdefault(record["pixel_sha256"], []).append(rid)
        if record.get("video_id"):
            by_video.setdefault(str(record["video_id"]), []).append(rid)
        if record.get("plate_identity"):
            by_plate.setdefault(str(record["plate_identity"]), []).append(rid)
    for bucket in (by_file, by_pixel, by_video, by_plate):
        for members in bucket.values():
            head = members[0]
            for other in members[1:]:
                uf.union(head, other)

    audits = []
    hashes = [(str(r["sample_id"]), int(r["ahash"])) for r in records]
    for i, (left_id, left_hash) in enumerate(hashes):
        for right_id, right_hash in hashes[i + 1 :]:
            if uf.find(left_id) == uf.find(right_id):
                continue
            dist = hamming(left_hash, right_hash)
            if dist <= phash_audit_max:
                audits.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "ahash_hamming": dist,
                        "merged": False,
                        "reason": "phash_is_not_sufficient_to_merge",
                    }
                )
    groups: dict[str, list[str]] = {}
    for record in records:
        rid = str(record["sample_id"])
        groups.setdefault(uf.find(rid), []).append(rid)
    return {"groups": groups, "phash_audit": audits}


def image_record(sample_id: str, path: Path, **extra: Any) -> dict[str, Any]:
    image = canonical_image(path)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "sample_id": sample_id,
        "file_sha256": sha256_file(path),
        "pixel_sha256": pixel_sha256(image),
        "ahash": average_hash(image),
        "width": image.size[0],
        "height": image.size[1],
        **extra,
    }
