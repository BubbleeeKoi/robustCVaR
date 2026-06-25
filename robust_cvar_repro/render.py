"""Rendering helpers for reproduced Figure 1 panels.

This module converts value functions, obstacle positions, and extracted
display paths into PNG comparison figures. Rendering is only a visualization
layer; algorithmic reproduction is handled by value_iteration.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "outputs" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from .gridworld import GridWorld


def render_comparison(
    env: GridWorld,
    panels: list[tuple[str, np.ndarray, list[tuple[int, int]], float]],
    output_path: Path,
    cell_size: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_images = [_render_panel(env, title, value, path, y, cell_size) for title, value, path, y in panels]

    margin = 24
    title_h = 24
    w = env.width * cell_size
    h = env.height * cell_size + title_h
    canvas = Image.new("RGB", (2 * w + 3 * margin, 2 * h + 3 * margin), "white")

    positions = [
        (margin, margin),
        (2 * margin + w, margin),
        (margin, 2 * margin + h),
        (2 * margin + w, 2 * margin + h),
    ]
    for img, pos in zip(panel_images, positions):
        canvas.paste(img, pos)

    canvas.save(output_path)


def render_paper_style_comparison(
    env: GridWorld,
    panels: list[tuple[str, np.ndarray, list[tuple[int, int]], float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, (title, values, path, _) in zip(axes.flat, panels):
        grid = values.reshape(env.height, env.width).copy()
        for x, y in env.obstacles:
            grid[y, x] = 60.0
        im = ax.imshow(grid, origin="upper", cmap="viridis", vmin=0.0, vmax=60.0)
        if path:
            xs = [xy[0] for xy in path]
            ys = [xy[1] for xy in path]
            ax.plot(xs, ys, color="red", linewidth=1.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, env.width - 1)
        ax.set_ylim(env.height - 1, 0)
        ax.set_xticks(np.arange(0, env.width + 1, 10))
        ax.set_yticks(np.arange(0, env.height + 1, 10))
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_panel(
    env: GridWorld,
    title: str,
    values: np.ndarray,
    path: list[tuple[int, int]],
    start_y: float,
    cell_size: int,
) -> Image.Image:
    title_h = 24
    w = env.width * cell_size
    h = env.height * cell_size
    img = Image.new("RGB", (w, h + title_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), title, fill=(20, 20, 20), font=ImageFont.load_default())

    finite = values[np.isfinite(values)]
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    denom = max(hi - lo, 1e-9)

    for state in range(env.n_states):
        x, y = env.to_xy(state)
        box = (x * cell_size, title_h + y * cell_size, (x + 1) * cell_size, title_h + (y + 1) * cell_size)
        if (x, y) in env.obstacles:
            color = (35, 35, 35)
        elif state == env.goal_state:
            color = (26, 140, 65)
        else:
            t = min(max((float(values[state]) - lo) / denom, 0.0), 1.0)
            color = _blue_yellow(t)
        draw.rectangle(box, fill=color)

    if len(path) > 1:
        points = [
            (x * cell_size + cell_size // 2, title_h + y * cell_size + cell_size // 2)
            for x, y in path
        ]
        draw.line(points, fill=(210, 20, 20), width=max(2, cell_size // 3))

    sx, sy = env.config.start
    gx, gy = env.config.goal
    draw.ellipse(
        (
            sx * cell_size + 1,
            title_h + sy * cell_size + 1,
            (sx + 1) * cell_size - 1,
            title_h + (sy + 1) * cell_size - 1,
        ),
        fill=(255, 255, 255),
        outline=(210, 20, 20),
        width=2,
    )
    draw.rectangle(
        (
            gx * cell_size + 1,
            title_h + gy * cell_size + 1,
            (gx + 1) * cell_size - 1,
            title_h + (gy + 1) * cell_size - 1,
        ),
        fill=(20, 160, 70),
        outline=(255, 255, 255),
    )
    return img


def _blue_yellow(t: float) -> tuple[int, int, int]:
    blue = np.array([36, 106, 178], dtype=float)
    yellow = np.array([247, 220, 90], dtype=float)
    rgb = (1.0 - t) * blue + t * yellow
    return tuple(int(v) for v in rgb)
