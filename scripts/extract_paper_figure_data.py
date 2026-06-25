"""Extract Figure 1 obstacle and path targets from the paper PDF.

The script ensures the four embedded Figure 1 panel images are available,
extracting them from PDF page 5 with PyMuPDF when needed. It then maps yellow
obstacle pixels and red path pixels onto the 64 x 53 GridWorld coordinate
system and writes CSV targets under outputs/.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PDF_IMAGE_DIR = ROOT / "outputs" / "pdf_images"
OUTPUTS = ROOT / "outputs"
PDF_PATH = ROOT / "Robust_Risk-Sensitive_Reinforcement_Learning_with_Conditional_Value-at-Risk.pdf"


PANEL_FILES = {
    "a_no_uncertainty": "page5_image3.png",
    "b_rn_k_2": "page5_image1.png",
    "c_kl_k_2": "page5_image2.png",
    "d_unfix_kappa": "page5_image0.png",
}


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    ensure_pdf_panel_images()
    obstacles = extract_obstacles(PDF_IMAGE_DIR / PANEL_FILES["a_no_uncertainty"])
    write_xy(OUTPUTS / "paper_extracted_obstacles.csv", sorted(obstacles), "x,y")
    print(f"extracted {len(obstacles)} obstacles")

    for slug, filename in PANEL_FILES.items():
        path = extract_red_path(PDF_IMAGE_DIR / filename)
        write_xy(OUTPUTS / f"paper_extracted_path_{slug}.csv", path, "x,y")
        print(f"extracted {len(path)} path points for {slug}")


def ensure_pdf_panel_images() -> None:
    missing = [
        PDF_IMAGE_DIR / filename
        for filename in PANEL_FILES.values()
        if not (PDF_IMAGE_DIR / filename).exists()
    ]
    if not missing:
        return

    try:
        import fitz
    except ImportError as exc:
        missing_files = ", ".join(path.name for path in missing)
        raise RuntimeError(
            "Figure 1 panel images are missing and PyMuPDF is not installed. "
            f"Install pymupdf or provide: {missing_files}"
        ) from exc

    PDF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with fitz.open(PDF_PATH) as doc:
        page = doc[4]
        images = page.get_images(full=True)
        if len(images) != 4:
            raise RuntimeError(f"Expected 4 embedded Figure 1 images on page 5, found {len(images)}")
        for index, image in enumerate(images):
            xref = image[0]
            pixmap = fitz.Pixmap(doc, xref)
            if pixmap.alpha or pixmap.n > 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            pixmap.save(PDF_IMAGE_DIR / f"page5_image{index}.png")


def extract_obstacles(image_path: Path) -> set[tuple[int, int]]:
    im = np.asarray(Image.open(image_path).convert("RGB"))
    mask = (im[:, :, 0] > 170) & (im[:, :, 1] > 170) & (im[:, :, 2] < 120)
    components = connected_components(mask)
    coords: set[tuple[int, int]] = set()
    for pts in components:
        if len(pts) < 4:
            continue
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        if cx >= 300 or cy <= 20:
            continue
        coords.add(pixel_to_grid(cx, cy))
    return coords


def extract_red_path(image_path: Path) -> list[tuple[int, int]]:
    im = np.asarray(Image.open(image_path).convert("RGB"))
    mask = (im[:, :, 0] > 180) & (im[:, :, 1] < 80) & (im[:, :, 2] < 80)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    grid_pts = sorted({pixel_to_grid(float(x), float(y)) for x, y in zip(xs, ys)})
    return order_path(grid_pts, start=(60, 50))


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            pts: list[tuple[int, int]] = []
            while q:
                px, py = q.popleft()
                pts.append((px, py))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            components.append(pts)
    return components


def pixel_to_grid(px: float, py: float) -> tuple[int, int]:
    left, right, top, bottom = 28.0, 291.0, 22.0, 246.0
    x = int(round((px - left) / (right - left) * 63.0))
    y = int(round((py - top) / (bottom - top) * 52.0))
    return max(0, min(63, x)), max(0, min(52, y))


def order_path(points: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    remaining = set(points)
    if start not in remaining:
        start = min(remaining, key=lambda xy: abs(xy[0] - start[0]) + abs(xy[1] - start[1]))
    ordered = [start]
    remaining.remove(start)
    while remaining:
        current = ordered[-1]
        nxt = min(remaining, key=lambda xy: abs(xy[0] - current[0]) + abs(xy[1] - current[1]))
        if abs(nxt[0] - current[0]) + abs(nxt[1] - current[1]) > 8:
            break
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def write_xy(path: Path, rows: list[tuple[int, int]], header: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y in rows:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    main()
