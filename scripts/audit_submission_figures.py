"""Audit submission-ready figure files for journal size and raster metadata."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image


EXPECTED_PIXELS = {
    "Fig1": (4110, 1800),
    "Fig2": (4110, 5520),
    "Fig3": (4110, 2490),
    "Fig4": (4110, 5400),
}
TARGET_DPI = 600
TARGET_WIDTH_MM = 174.0
MAX_HEIGHT_MM = 234.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory: Path) -> None:
    errors: list[str] = []
    for stem, expected_size in EXPECTED_PIXELS.items():
        for suffix in (".png", ".tiff"):
            path = directory / f"{stem}{suffix}"
            if not path.exists():
                errors.append(f"missing: {path.name}")
                continue
            with Image.open(path) as image:
                dpi = image.info.get("dpi")
                if image.size != expected_size:
                    errors.append(
                        f"{path.name}: pixels {image.size} != {expected_size}"
                    )
                if image.mode != "RGB":
                    errors.append(f"{path.name}: mode {image.mode} != RGB")
                if not dpi or any(abs(value - TARGET_DPI) > 0.1 for value in dpi):
                    errors.append(f"{path.name}: dpi {dpi} != {TARGET_DPI}")
                    continue
                width_mm = image.size[0] / dpi[0] * 25.4
                height_mm = image.size[1] / dpi[1] * 25.4
                if abs(width_mm - TARGET_WIDTH_MM) > 0.05:
                    errors.append(
                        f"{path.name}: width {width_mm:.2f} mm != {TARGET_WIDTH_MM:.2f} mm"
                    )
                if height_mm > MAX_HEIGHT_MM:
                    errors.append(
                        f"{path.name}: height {height_mm:.2f} mm > {MAX_HEIGHT_MM:.2f} mm"
                    )
                dpi_x = float(dpi[0])
                print(
                    f"{path.name}: {image.size[0]}x{image.size[1]} px, "
                    f"{image.mode}, {dpi_x:.1f} dpi, "
                    f"{width_mm:.2f}x{height_mm:.2f} mm, "
                    f"sha256={sha256(path)}"
                )

    if errors:
        raise SystemExit("Figure audit failed:\n- " + "\n- ".join(errors))
    print("Figure audit passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission_figures",
    )
    audit(parser.parse_args().directory)
