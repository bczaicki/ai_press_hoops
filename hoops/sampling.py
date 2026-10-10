"""Where to look in a clip: uniform overviews, CV-proposed event windows, motion peaks, scene cuts.

All times are seconds on the master timeline (the proxy shares t=0). Read normalized files only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def uniform_times(duration_s: float, fps: float, max_n: int | None = None, t0: float = 0.0) -> list[float]:
    """Evenly spaced times in [t0, duration_s) at ``fps``.

    When that would exceed ``max_n`` samples, return ``max_n`` times spread over the same span instead.
    """
    span = duration_s - t0
    if span <= 0 or fps <= 0:
        return []
    n = max(1, math.ceil(span * fps - 1e-9))
    step = 1.0 / fps
    if max_n is not None and n > max_n:
        if max_n <= 0:
            return []
        n, step = max_n, span / max_n
    return [round(t0 + i * step, 6) for i in range(n)]


def window_times(t0: float, t1: float, fps: float) -> list[float]:
    """Times in [t0, t1) at ``fps`` (e.g. 4-8 fps over a 3.5 s candidate window)."""
    return uniform_times(t1, fps, t0=t0)


def event_windows(
    proposals_s: Sequence[float],
    pre: float = 2.0,
    post: float = 1.5,
    merge_gap: float = 0.5,
    duration_s: float | None = None,
) -> list[tuple[float, float]]:
    """Expand proposal times to [t - pre, t + post] windows, merging windows closer than ``merge_gap``."""
    out: list[list[float]] = []
    for t in sorted(proposals_s):
        a, b = max(0.0, t - pre), t + post
        if duration_s is not None:
            b = min(b, duration_s)
            if a >= b:
                continue
        if out and a <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 6), round(b, 6)) for a, b in out]


def motion_peaks(path: Path, sample_fps: float = 10, top_k: int = 20, min_gap_s: float = 1.0) -> list[float]:
    """Times of the largest frame-to-frame changes (cheap proposals for fixed cameras), sorted by time."""
    import av

    times: list[float] = []
    scores: list[float] = []
    prev: np.ndarray | None = None
    next_t = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        w = 160
        h = max(2, round(w * stream.codec_context.height / max(1, stream.codec_context.width)))
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < next_t:
                continue
            next_t = frame.time + 1.0 / sample_fps
            gray = frame.reformat(width=w, height=h, format="gray").to_ndarray().astype(np.int16)
            if prev is not None:
                times.append(float(frame.time))
                scores.append(float(np.abs(gray - prev).mean()))
            prev = gray
    picked: list[int] = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] <= 0:
            break
        if all(abs(times[i] - times[j]) >= min_gap_s for j in picked):
            picked.append(int(i))
            if len(picked) == top_k:
                break
    return sorted(round(times[i], 3) for i in picked)


def scene_cuts(path: Path, threshold: float = 3.0, min_scene_len_s: float = 0.6) -> list[float]:
    """Hard-cut times in seconds (PySceneDetect AdaptiveDetector). Empty for continuous footage."""
    from scenedetect import AdaptiveDetector, detect

    detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=f"{min_scene_len_s}s")
    try:
        scenes = detect(str(path), detector, backend="pyav")
    except Exception:  # noqa: BLE001 - fall back to the OpenCV reader if PyAV trips on a file
        scenes = detect(str(path), detector, backend="opencv")
    return [round(start.seconds, 3) for start, _ in scenes[1:]]
