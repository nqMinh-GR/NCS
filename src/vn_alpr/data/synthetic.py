"""Small synthetic plates for pipeline self-tests. Not a benchmark score."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def write_plate_image(path: Path, text: str, width: int = 160, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, width - 3, height - 3), outline=(20, 20, 20), width=2)
    draw.text((8, 8), text, fill=(10, 10, 10), font=_font(18))
    image.save(path)


def make_synthetic_detection(root: Path, n: int = 4) -> list[dict]:
    samples = []
    for index in range(n):
        text = f"51A{12340 + index}"
        path = root / "images" / f"syn_{index}.png"
        write_plate_image(path, text)
        samples.append(
            {
                "sample_id": f"syn_{index}",
                "file_name": path.name,
                "image_path": str(path),
                "width": 160,
                "height": 48,
                "dataset_id": "synthetic_vn",
                "source_family_id": "synthetic_vn",
                "boxes": [{"name": "license_plate", "bbox": [4, 4, 150, 40]}],
                "text_raw": text,
                "layout": "single_line",
                "country": "VN",
            }
        )
    return samples
