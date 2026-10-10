"""Frame access on normalized videos (PyAV).

Frames are RGB uint8 arrays (H, W, 3). Time is seconds on the file's timeline (the master and the
proxy share t=0). Only read normalized files (master/proxy): they are upright, SDR, constant frame rate.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from pathlib import Path

import av
import numpy as np
from PIL import Image


def iter_frames(
    path: Path | str,
    t0: float | None = None,
    t1: float | None = None,
    fps: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_s, rgb) for frames in [t0, t1). With ``fps``, keep roughly that many frames per second."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # ~5x faster than the default SLICE threading
        if t0 and t0 > 0:
            container.seek(int(t0 * av.time_base), backward=True, any_frame=False)  # keyframe <= t0
        step = 1.0 / fps if fps else 0.0
        next_t = t0 or 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if t0 is not None and t < t0 - 1e-6:
                continue
            if t1 is not None and t >= t1:
                break
            if step:
                if t + 1e-6 < next_t:
                    continue
                while next_t <= t + 1e-6:
                    next_t += step
            yield t, frame.to_ndarray(format="rgb24")


def frames_at(path: Path | str, times_s: Sequence[float]) -> list[tuple[float, np.ndarray]]:
    """Return the frame nearest each requested time (one decode pass), in the order requested."""
    if not times_s:
        return []
    order = sorted(range(len(times_s)), key=lambda i: times_s[i])
    targets = [times_s[i] for i in order]
    best: list[tuple[float, float, np.ndarray | None]] = [(float("inf"), 0.0, None) for _ in targets]
    lo, hi = max(0.0, targets[0] - 0.5), targets[-1] + 0.5
    for t, rgb in iter_frames(path, t0=lo, t1=hi):
        for j, target in enumerate(targets):
            d = abs(t - target)
            if d < best[j][0]:
                best[j] = (d, t, rgb)
        if t > targets[-1] and all(b[2] is not None and b[0] < 0.5 for b in best):
            break
    out: list[tuple[float, np.ndarray] | None] = [None] * len(times_s)
    for j, i in enumerate(order):
        _, t, rgb = best[j]
        if rgb is not None:
            out[i] = (t, rgb)
    return [x for x in out if x is not None]


def to_jpeg(rgb: np.ndarray, max_px: int = 1280, quality: int = 85) -> bytes:
    """Encode an RGB frame as JPEG, downscaling so the long edge is at most ``max_px``."""
    img = Image.fromarray(rgb)
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop(
    rgb: np.ndarray,
    box: tuple[float, float, float, float],
    pad: float = 0.25,
    min_size: int = 64,
) -> np.ndarray:
    """Crop (x1, y1, x2, y2) with relative padding, clamped to the frame and at least ``min_size`` px."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = max((x2 - x1) * (1 + 2 * pad), min_size)
    bh = max((y2 - y1) * (1 + 2 * pad), min_size)
    left, top = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    right, bottom = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
    return rgb[top:bottom, left:right].copy()
