#!/usr/bin/env python3
"""
heatmap_text_writer.py

Generate a numeric heatmap and embed text with deliberately lo‑fi aesthetics.

What is new in *this* revision
-----------------------------
* **Sparse noise**. `create_canvas` now has a `noise_density` knob, letting you
  sprinkle bright speckles instead of washing the whole map in uniform grain.
  • `noise_density = 1.0` (default) reproduces the old behaviour.
  • `noise_density = 0.05` lights up ~5 % of the pixels, giving classic heat‑
    map activations.
* The original `pixel_size`, `jitter`, and `seed` controls all remain.

Public helpers
--------------
create_canvas(height, width, noise_level=0.1, noise_density=1.0, seed=None)
    Return a noisy 2‑D NumPy array.  Lower `noise_density` → sparser, brighter
    speckles.

write_text(canvas, text, *, font_size, xy=(0,0), intensity=1.0,
           pixel_size=1, jitter=0.0, font_path=None, seed=None)
    Stamp *text* into *canvas* **in‑place**, using `pixel_size` to pixelate and
    `jitter` to vary brightness.

show_canvas(canvas)
    Quick matplotlib preview with a colour‑bar.

Example
-------
>>> canvas = create_canvas(200, 320, noise_level=0.15, noise_density=0.04, seed=3)
>>> write_text(canvas, "Sparse!", font_size=56, xy=(30, 60),
>>>            intensity=1.4, pixel_size=5, jitter=0.4, seed=7)
>>> show_canvas(canvas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "create_canvas",
    "write_text",
    "show_canvas",
]

# -----------------------------------------------------------------------------
# Canvas utilities
# -----------------------------------------------------------------------------


def create_canvas(
    height: int,
    width: int,
    *,
    noise_level: float = 0.1,
    noise_density: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return a *height*×*width* canvas populated with random noise.

    Parameters
    ----------
    height, width : int
        Canvas dimensions in pixels.
    noise_level : float, optional
        Maximum value of the uniform noise distribution (upper bound).
    noise_density : float, optional
        Fraction of pixels that receive non‑zero noise. 1.0 means every pixel
        gets a sample, lower values yield sparser activations.
    seed : int, optional
        Seed for reproducible noise.
    """
    if not (0.0 < noise_density <= 1.0):
        raise ValueError("noise_density must be in (0, 1]")

    rng = np.random.default_rng(seed)

    if noise_density == 1.0:
        # Original behaviour: fully distributed noise.
        return rng.uniform(0.0, noise_level, size=(height, width)).astype(np.float32)

    # Sparse speckles.
    canvas = np.zeros((height, width), dtype=np.float32)
    mask = rng.random((height, width)) < noise_density  # Boolean mask.
    canvas[mask] = rng.uniform(0.0, noise_level, size=mask.sum())
    return canvas


# -----------------------------------------------------------------------------
# Text stamping helper
# -----------------------------------------------------------------------------


def _default_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Return DejaVuSans (bundled with matplotlib) or PIL default bitmap font."""
    try:
        path = Path(ImageFont.__file__).with_name("DejaVuSans.ttf")
        return ImageFont.truetype(str(path), font_size)
    except Exception:
        return ImageFont.load_default()


def write_text(
    canvas: np.ndarray,
    text: str,
    *,
    font_size: int,
    xy: Tuple[int, int] = (0, 0),
    intensity: float = 1.0,
    pixel_size: int = 1,
    jitter: float = 0.0,
    font_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Embed *text* inside *canvas* with retro pixel‑art vibes.

    All arguments behave exactly like the previous version, with *no changes*.
    Consult the top‑level docstring for a quick refresher.
    """
    if canvas.ndim != 2:
        raise ValueError("canvas must be a 2‑D array")
    if pixel_size < 1:
        raise ValueError("pixel_size must be ≥ 1")

    height, width = canvas.shape
    rng = np.random.default_rng(seed)

    # 1. Render the text into a high‑res mask.
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path is not None
        else _default_font(font_size)
    )
    draw.text(xy, text, fill=255, font=font)

    # 2. Pixelate mask via down‑ and up‑scaling.
    if pixel_size > 1:
        small_w = max(1, width // pixel_size)
        small_h = max(1, height // pixel_size)
        mask_small = mask_img.resize((small_w, small_h), Image.Resampling.BOX)
        mask_img = mask_small.resize((width, height), Image.Resampling.NEAREST)

    mask = np.asarray(mask_img, dtype=np.float32) / 255.0  # Range 0‑1.

    # 3. Apply intensity plus per‑pixel jitter.
    base = mask * intensity
    if jitter > 0.0:
        jitter_noise = rng.uniform(-jitter, jitter, size=mask.shape) * intensity
        base += jitter_noise * mask  # only where glyph exists.

    # 4. Update canvas.
    canvas += base
    return canvas


# -----------------------------------------------------------------------------
# Quick visualisation helper
# -----------------------------------------------------------------------------


def show_canvas(canvas: np.ndarray) -> None:
    """Display the heatmap with a colour‑bar using matplotlib."""
    plt.figure(figsize=(6, 6 * canvas.shape[0] / canvas.shape[1]))
    plt.imshow(canvas, origin="upper")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("heatmap_text.png", dpi=300, bbox_inches="tight", pad_inches=0)


# -----------------------------------------------------------------------------
# Self‑test demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    r = 2
    H, W = int(396 / r), int(1584 / r)

    # Sparse, speckly background.
    canvas = create_canvas(
        H,
        W,
        noise_level=0.9,
        noise_density=0.18,  # about 3 % of pixels lit.
        seed=42,
    )

    # Chunky pixel text on top.
    write_text(
        canvas,
        "Passion is all you need.",
        font_size=48,
        xy=(W / 3.65, H / 3),
        intensity=1.5,
        pixel_size=6,
        jitter=0.35,
        seed=99,
    )
    write_text(
        canvas,
        "Marwan Mashra",
        font_size=20,
        xy=(W - W / 4.4, H - H / 5.5),
        intensity=1.5,
        pixel_size=2,
        jitter=0.35,
        seed=99,
    )
    write_text(
        canvas,
        "gently cooked by",
        font_size=16,
        xy=(W - W / 2.46, H - H / 6),
        intensity=1.3,
        pixel_size=2,
        jitter=0.35,
        seed=99,
    )

    show_canvas(canvas)
