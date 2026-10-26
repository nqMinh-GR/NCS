"""Plate text normalization. Confusion pairs are never rewritten."""

from __future__ import annotations

import re
import unicodedata

SEPARATORS = set(" -._·•/\\")


def normalize_plate(text: str | None) -> str:
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).upper()
    kept = []
    for char in folded:
        if char in SEPARATORS or char.isspace():
            continue
        kept.append(char)
    return "".join(kept)


def unknown_characters(text: str, alphabet: str) -> list[str]:
    """Characters that normalization kept but the model alphabet does not contain."""
    allowed = set(alphabet)
    return [char for char in normalize_plate(text) if char not in allowed]


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def looks_like_vn_plate(normalized: str, pattern: str) -> bool:
    return re.fullmatch(pattern, normalized) is not None
