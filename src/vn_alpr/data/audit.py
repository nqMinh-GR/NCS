"""Contact-sheet audit images and class counts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def contact_sheet(image_paths: list[Path], dest: Path, columns: int = 4, thumb: int = 160) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        image = Image.new("RGB", (thumb, thumb), (255, 255, 255))
        image.save(dest)
        return dest
    rows = (len(image_paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb, rows * thumb), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(image_paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb - 8, thumb - 8))
            x = (index % columns) * thumb + 4
            y = (index // columns) * thumb + 4
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + image.size[0], y + image.size[1]), outline=(0, 0, 0))
    canvas.save(dest)
    return dest


def class_histogram(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        for box in sample.get("boxes", []):
            counts[str(box.get("name"))] += 1
    return dict(counts)
