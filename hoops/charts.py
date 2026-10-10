"""Court charts as PNGs (Pillow + numpy): shot chart, court heatmap, image-space heatmap fallback.

Dark broadcast style. Made = filled green circle, missed = red X: shape carries the meaning as well as
color (the pair is only moderately separable for red-green colorblind viewers). Heatmaps use one blue
ramp, dark -> light, so denser reads brighter on the dark surface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hoops.perception.court import court_lines, fold_to_half, spec

SURFACE = (26, 26, 25)  # #1a1a19
FLOOR = (36, 36, 33)
LINE = (195, 194, 183)  # #c3c2b7, also secondary text
TEXT = (255, 255, 255)
MADE = (12, 163, 12)  # #0ca30c
MISSED = (230, 103, 103)  # #e66767
RAMP = np.array(  # blue 650 -> 100
    [(16, 66, 129), (28, 92, 171), (42, 120, 214), (85, 152, 231), (134, 182, 239), (205, 226, 251)], float
)
SS = 2  # supersampling factor (Pillow lines aren't anti-aliased)
PX_PER_FT = 12
MARGIN, HEADER, FOOTER = 28, 76, 36


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size * SS)


def _layout(court: str, half: bool) -> tuple[int, int, Callable[[np.ndarray], np.ndarray]]:
    """Canvas size (final px) and a court-ft -> canvas-px transform (at SS scale)."""
    c = spec(court)
    hl, hw, s = c.length_ft / 2, c.width_ft / 2, PX_PER_FT
    if half:
        w, h = round(c.width_ft * s), round(hl * s)

        def tf(p: np.ndarray) -> np.ndarray:  # basket at the top, camera-left on the left
            return np.c_[MARGIN + (hw - p[:, 1]) * s, HEADER + (hl - p[:, 0]) * s] * SS

    else:
        w, h = round(c.length_ft * s), round(c.width_ft * s)

        def tf(p: np.ndarray) -> np.ndarray:
            return np.c_[MARGIN + (p[:, 0] + hl) * s, HEADER + (hw - p[:, 1]) * s] * SS

    return w + 2 * MARGIN, h + HEADER + FOOTER, tf


def _canvas(court: str, half: bool) -> tuple[Image.Image, Callable[[np.ndarray], np.ndarray]]:
    w, h, tf = _layout(court, half)
    img = Image.new("RGB", (w * SS, h * SS), SURFACE)
    box = tf(court_lines(court, half=half)[0])  # outline
    ImageDraw.Draw(img).rectangle([*box.min(0), *box.max(0)], fill=FLOOR)
    return img, tf


def _draw_lines(img: Image.Image, court: str, half: bool, tf: Callable[[np.ndarray], np.ndarray]) -> None:
    draw = ImageDraw.Draw(img)
    for poly in court_lines(court, half=half):
        draw.line([tuple(p) for p in tf(poly)], fill=LINE, width=2 * SS, joint="curve")


def _header(img: Image.Image, title: str, subtitle: str | None = None) -> None:
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN * SS, 14 * SS), title, fill=TEXT, font=_font(22))
    if subtitle:
        draw.text((MARGIN * SS, 44 * SS), subtitle, fill=LINE, font=_font(15))


def _footer(img: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN * SS, img.height - 26 * SS), text, fill=LINE, font=_font(13))


def _save(img: Image.Image, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.resize((img.width // SS, img.height // SS), Image.Resampling.LANCZOS).save(out_path, "PNG")
    return out_path


def _made_marker(draw: ImageDraw.ImageDraw, x: float, y: float, r: float = 7) -> None:
    r, ring = r * SS, 2 * SS
    draw.ellipse([x - r - ring, y - r - ring, x + r + ring, y + r + ring], fill=SURFACE)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=MADE)


def _missed_marker(draw: ImageDraw.ImageDraw, x: float, y: float, r: float = 6) -> None:
    r = r * SS
    for w, color in ((5 * SS, SURFACE), (3 * SS, MISSED)):
        draw.line([(x - r, y - r), (x + r, y + r)], fill=color, width=w)
        draw.line([(x - r, y + r), (x + r, y - r)], fill=color, width=w)


def shot_chart(
    shots: Sequence[tuple[float, float, bool]],
    out_path: Path,
    *,
    court: str = "nba",
    half: bool = True,
    title: str | None = None,
) -> Path:
    """Plot (x_ft, y_ft, made) shots. With ``half``, shots on the far half are folded onto the near one."""
    img, tf = _canvas(court, half)
    _draw_lines(img, court, half, tf)
    pts = np.asarray([(x, y) for x, y, _ in shots], float).reshape(-1, 2)
    made = np.asarray([bool(m) for *_, m in shots], bool)
    if half and len(pts):
        pts = fold_to_half(pts)
    draw = ImageDraw.Draw(img)
    px = tf(pts) if len(pts) else np.zeros((0, 2))
    for (x, y), m in sorted(zip(px.tolist(), made.tolist(), strict=True), key=lambda t: t[1]):  # makes on top
        (_made_marker if m else _missed_marker)(draw, x, y)
    n, k = len(made), int(made.sum())
    subtitle = f"{k}/{n} made ({k / n:.0%})" if n else "No shots to show yet"
    _header(img, title or "Shot chart", subtitle)
    # Legend, right-aligned in the header: shapes as well as colors.
    font = _font(14)
    x = img.width - MARGIN * SS
    for label, marker in ((f"Missed {n - k}", _missed_marker), (f"Made {k}", _made_marker)):
        tw = draw.textlength(label, font=font)
        x -= tw
        draw.text((x, 47 * SS), label, fill=LINE, font=font)
        marker(draw, x - 14 * SS, 56 * SS, 6)
        x -= 40 * SS
    _footer(img, f"{spec(court).label} court · shots at the shooter's feet")
    return _save(img, out_path)


def _colorize(t: np.ndarray) -> np.ndarray:
    """[0, 1] density -> RGBA uint8 (transparent where there's nothing)."""
    idx = np.clip(t, 0, 1) * (len(RAMP) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(RAMP) - 1)
    frac = (idx - lo)[..., None]
    rgb = RAMP[lo] * (1 - frac) + RAMP[hi] * frac
    alpha = np.where(t > 0.03, np.clip(t, 0, 1) ** 0.6 * 230, 0)
    return np.dstack([rgb, alpha]).astype(np.uint8)


def _blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, grid)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, out)


def _density_layer(
    px: np.ndarray, size: tuple[int, int], bin_px: float, sigma_bins: float = 1.5
) -> Image.Image:
    """RGBA heat layer of ``size`` (w, h) from pixel positions."""
    w, h = size
    nx, ny = max(1, round(w / bin_px)), max(1, round(h / bin_px))
    grid, _, _ = np.histogram2d(px[:, 1], px[:, 0], bins=(ny, nx), range=((0, h), (0, w)))
    grid = _blur(grid, sigma_bins)
    if grid.max() > 0:
        grid /= grid.max()
    layer = Image.fromarray(_colorize(grid), "RGBA")
    return layer.resize((w, h), Image.Resampling.BILINEAR)


def _gradient_key(img: Image.Image, x: int, y: int, w: int = 120, h: int = 10) -> None:
    bar = Image.fromarray(_colorize(np.tile(np.linspace(0.05, 1, w * SS), (h * SS, 1))), "RGBA")
    img.paste(bar, (x, y), bar)
    draw, font = ImageDraw.Draw(img), _font(12)
    draw.text((x - draw.textlength("fewer", font=font) - 6 * SS, y - 4 * SS), "fewer", fill=LINE, font=font)
    draw.text((x + (w + 6) * SS, y - 4 * SS), "more", fill=LINE, font=font)


def court_heatmap(
    points_ft: np.ndarray,
    out_path: Path,
    *,
    court: str = "nba",
    half: bool = False,
    title: str | None = None,
) -> Path:
    """Density of court positions (feet), e.g. players' feet over a run."""
    img, tf = _canvas(court, half)
    pts = np.asarray(points_ft, float).reshape(-1, 2)
    if half and len(pts):
        pts = fold_to_half(pts)
    c = spec(court)
    if len(pts):
        inside = (np.abs(pts[:, 0]) <= c.length_ft / 2 + 3) & (np.abs(pts[:, 1]) <= c.width_ft / 2 + 3)
        layer = _density_layer(tf(pts[inside]), img.size, bin_px=PX_PER_FT * SS)
        img.paste(layer, (0, 0), layer)
    _draw_lines(img, court, half, tf)
    _header(img, title or "Where the action was", f"{len(pts)} positions" if len(pts) else "No positions yet")
    _gradient_key(img, img.width - (MARGIN + 160) * SS, 50 * SS)
    _footer(img, f"{c.label} court · positions mapped with the court calibration")
    return _save(img, out_path)


def image_heatmap(
    points_px: np.ndarray, background_rgb: np.ndarray, out_path: Path, *, title: str | None = None
) -> Path:
    """Fallback without calibration: density over a video frame, labeled as camera (image) space."""
    bg = Image.fromarray(np.asarray(background_rgb, np.uint8)).convert("RGB")
    bg = Image.eval(bg, lambda v: int(v * 0.5))  # dim the frame so the heat reads
    w, h = bg.size
    canvas = Image.new("RGB", (w * SS, (h + HEADER) * SS), SURFACE)
    canvas.paste(bg.resize((w * SS, h * SS), Image.Resampling.BILINEAR), (0, HEADER * SS))
    pts = np.asarray(points_px, float).reshape(-1, 2)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts):
        layer = _density_layer(pts * SS, (w * SS, h * SS), bin_px=max(w, h) / 48 * SS)
        canvas.paste(layer, (0, HEADER * SS), layer)
    _header(
        canvas,
        title or "Where the action was (image-space)",
        "Camera view, not court feet (no court calibration)",
    )
    return _save(canvas, out_path)
