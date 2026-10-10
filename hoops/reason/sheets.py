"""VLM inputs from video: contact sheets with burned-in tile ids, window sheets, and overview sheets.

Each tile carries a yellow-on-black chip like ``S2#4 00:12.40``. Models cite the tile id; code maps it back to
seconds with ``resolve_ref``. Frames are RGB uint8; times are seconds on the file's timeline.
"""

from __future__ import annotations

import functools
import io
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hoops.frames import crop, iter_frames
from hoops.reason.base import ImagePart

TILE_RE = re.compile(r"\b([A-Za-z]{1,3}\d*#\d+)\b")
CLOCK_RE = re.compile(r"(?<![\d.])(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)(?![\d])")
SECONDS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")


def mmss(t: float) -> str:
    """Seconds -> 'MM:SS.ss' (what we burn into tiles)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def clock(t: float) -> str:
    """Seconds -> 'M:SS' for people."""
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


@functools.lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001 - very old Pillow / no FreeType
        return ImageFont.load_default()


def _fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Letterbox ``img`` into ``size`` (keeps aspect; black bars)."""
    tw, th = size
    scale = min(tw / img.width, th / img.height)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    out = Image.new("RGB", size, (0, 0, 0))
    out.paste(img.resize((w, h), Image.Resampling.LANCZOS), ((tw - w) // 2, (th - h) // 2))
    return out


def _chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int) -> None:
    font = _font(size)
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    pad = max(2, size // 5)
    draw.rectangle((l - pad, t - pad, r + pad, b + pad), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 230, 0))


def contact_sheet(
    frames_rgb: Sequence[np.ndarray],
    times_s: Sequence[float],
    sheet_id: str,
    cols: int = 3,
    tile: tuple[int, int] = (640, 360),
) -> tuple[Image.Image, dict[str, float]]:
    """Tile frames into one image; returns (sheet, {tile_id: seconds}). Tile ids are 1-based: 'S2#1'..."""
    if len(frames_rgb) != len(times_s):
        raise ValueError("frames_rgb and times_s must have the same length")
    if not frames_rgb:
        raise ValueError("contact_sheet needs at least one frame")
    cols = max(1, min(cols, len(frames_rgb)))
    rows = math.ceil(len(frames_rgb) / cols)
    tw, th = tile
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font_px = max(12, round(min(tw, th) * 0.075))
    index: dict[str, float] = {}
    for i, (rgb, t) in enumerate(zip(frames_rgb, times_s, strict=True)):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(_fit(Image.fromarray(rgb), tile), (x, y))
        tile_id = f"{sheet_id}#{i + 1}"
        index[tile_id] = round(float(t), 3)
        _chip(draw, (x + font_px // 2 + 2, y + font_px // 2 + 2), f"{tile_id} {mmss(t)}", font_px)
    for c in range(1, cols):  # thin separators so tiles read as separate frames
        draw.line([(c * tw, 0), (c * tw, rows * th)], fill=(255, 255, 255), width=2)
    for r in range(1, rows):
        draw.line([(0, r * th), (cols * tw, r * th)], fill=(255, 255, 255), width=2)
    return sheet, index


def resolve_ref(ref: str, index: dict[str, float]) -> float | None:
    """Tile id ('S2#4'), 'MM:SS(.s)', 'H:MM:SS', or plain seconds ('12.4s') -> seconds; None if unresolvable."""
    if not ref:
        return None
    s = str(ref).strip()
    if s in index:
        return index[s]
    upper = {k.upper(): v for k, v in index.items()}
    for m in TILE_RE.finditer(s):
        if m.group(1).upper() in upper:
            return upper[m.group(1).upper()]
    m = CLOCK_RE.search(s)
    if m:
        h, mins, secs = m.groups()
        return round((int(h) * 3600 if h else 0) + int(mins) * 60 + float(secs), 3)
    m = SECONDS_RE.match(s)
    if m:
        return float(m.group(1))
    return None


def sheet_to_part(sheet: Image.Image, index: dict[str, float], sheet_id: str, max_px: int) -> ImagePart:
    img = sheet.copy()
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    times = sorted(index.values())
    tiles = list(index)
    label = (
        f"Sheet {sheet_id}: tiles {tiles[0]}..{tiles[-1]} in time order, {mmss(times[0])}-{mmss(times[-1])}"
        if len(tiles) > 1
        else f"Frame {tiles[0]} at {mmss(times[0])}"
    )
    return ImagePart(
        jpeg=buf.getvalue(), label=label, t_s=times[0], tile_id=tiles[0] if len(tiles) == 1 else None
    )


def _sheets(
    frames: list[tuple[float, np.ndarray]],
    prefix: str,
    *,
    cols: int,
    rows: int,
    tile: tuple[int, int],
    max_px: int,
    start: int = 1,
) -> tuple[list[ImagePart], dict[str, float]]:
    parts: list[ImagePart] = []
    index: dict[str, float] = {}
    per = max(1, cols * rows)
    for k in range(0, len(frames), per):
        chunk = frames[k : k + per]
        sid = f"{prefix}{start + k // per}"
        sheet, idx = contact_sheet([f for _, f in chunk], [t for t, _ in chunk], sid, cols=cols, tile=tile)
        parts.append(sheet_to_part(sheet, idx, sid, max_px))
        index.update(idx)
    return parts, index


def window_images(
    video: Path,
    t0: float,
    t1: float,
    *,
    fps: float = 6.0,
    sheet_prefix: str = "W",
    crop_box: tuple | None = None,
    max_px: int = 1280,
) -> tuple[list[ImagePart], dict[str, float]]:
    """Dense sheets for one event window. With ``crop_box`` (x1, y1, x2, y2 in the video's pixels, e.g. the rim
    on the proxy), tiles are square crops around it; otherwise wide 16:9 frames."""
    frames = list(iter_frames(video, t0=max(0.0, t0), t1=t1, fps=fps))
    if not frames:
        return [], {}
    if crop_box is not None:
        # ~4 rim-widths of context so the ball's approach above the rim is in frame
        frames = [(t, crop(rgb, tuple(crop_box), pad=1.5, min_size=128)) for t, rgb in frames]
        side = max(160, max_px // 4)
        cols = max(1, max_px // side)
        return _sheets(frames, sheet_prefix, cols=cols, rows=cols, tile=(side, side), max_px=max_px)
    tw = min(640, max_px)
    tile = (tw, round(tw * 9 / 16))
    cols = max(1, max_px // tile[0])
    rows = max(1, min(cols, max_px // tile[1]))
    return _sheets(frames, sheet_prefix, cols=cols, rows=rows, tile=tile, max_px=max_px)


def _sample_times(duration_s: float, n: int) -> list[float]:
    step = duration_s / n
    return [round(step * (i + 0.5), 3) for i in range(n)]


def overview_images(
    video: Path,
    duration_s: float,
    *,
    fps: float = 1.0,
    max_sheets: int = 6,
) -> tuple[list[ImagePart], dict[str, float]]:
    """Whole-clip 3x3 sheets ('S1', 'S2', ...) of 640x360 tiles at ``fps`` (thinned to fit ``max_sheets``)."""
    if duration_s <= 0:
        return [], {}
    n = max(1, min(math.ceil(duration_s * fps), max_sheets * 9))
    times = _sample_times(duration_s, n)
    if duration_s / n >= 2.0:  # sparse: seek per sample instead of decoding the whole clip
        frames = []
        for t in times:
            got = next(iter_frames(video, t0=t, t1=t + 1.0), None)
            if got is not None:
                frames.append(got)
    else:
        frames = list(iter_frames(video, t0=0.0, t1=duration_s + 0.001, fps=n / duration_s))[:n]
    if not frames:
        return [], {}
    return _sheets(frames, "S", cols=3, rows=3, tile=(640, 360), max_px=1920)
