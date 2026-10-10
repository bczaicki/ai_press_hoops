"""Annotated video: broadcast-style overlays drawn on the proxy, written as browser-playable H.264.

Overlays: an ellipse under each player (team colour when known), a small track-id chip, a triangle over the
ball plus a ~0.5 s trail, a thin rim box (never filled, so the net stays readable), a timestamp chip, and a
brief MADE / MISS / SHOT flash around each shot event. supervision/OpenCV draw in BGR; frames are converted
at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from hoops.frames import iter_frames
from hoops.schemas import Event, EventType
from hoops.video_io import VideoWriter, mux_audio

NEUTRAL = "#6b7280"
REFEREE = "#111827"
DEFAULT_TEAM_COLORS = {"A": "#2563eb", "B": "#dc2626"}
BALL = "#f97316"
RIM = "#22d3ee"
FLASH_COLORS = {"MADE": "#16a34a", "MISS": "#dc2626", "SHOT": "#d97706"}
TRAIL_S = 0.5
FLASH_S = 0.75
PEOPLE = ("player", "referee", "person")
SHOT_TYPES = {EventType.shot_attempt, EventType.three_pointer, EventType.free_throw}


def _bgr(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def _text_on(hex_color: str) -> str:
    b, g, r = _bgr(hex_color)
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


def flash_label(event: Event) -> str | None:
    if event.status == "rejected":
        return None
    result = event.shot.result if event.shot else "unknown"
    if event.type == EventType.made_shot or (event.type in SHOT_TYPES and result == "made"):
        return "MADE"
    if event.type == EventType.missed_shot or (event.type in SHOT_TYPES and result == "missed"):
        return "MISS"
    return "SHOT" if event.type in SHOT_TYPES else None


def _clock(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _chip(
    img: np.ndarray, text: str, org: tuple[int, int], bg: str, scale: float, alpha: float = 0.75
) -> None:
    """Text on a translucent rounded-ish box; ``org`` is the box's top-left corner."""
    import cv2

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(2 * scale)))
    pad = int(8 * scale) + 4
    x, y = org
    x2, y2 = min(img.shape[1] - 1, x + tw + 2 * pad), min(img.shape[0] - 1, y + th + base + 2 * pad)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    roi[:] = (roi * (1 - alpha) + np.array(_bgr(bg)) * alpha).astype(np.uint8)
    cv2.putText(
        img,
        text,
        (x + pad, y + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _bgr(_text_on(bg)),
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )


def _video_info(path: Path) -> tuple[Fraction, int, int, int]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = s.average_rate or s.guessed_rate or Fraction(30)
        return Fraction(rate), s.codec_context.width, s.codec_context.height, int(s.frames or 0)


def render_annotated(
    proxy_path: Path,
    detections: pd.DataFrame,
    out_path: Path,
    *,
    events: Sequence[Event] = (),
    team_colors: dict[str, str] | None = None,
    rim_box: tuple | None = None,
    master_audio: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Draw overlays on every proxy frame and write an H.264/yuv420p/faststart MP4 (audio copied when given)."""
    import cv2
    import supervision as sv

    proxy_path, out_path = Path(proxy_path), Path(out_path)
    rate, width, height, n_frames = _video_info(proxy_path)

    colors = {**DEFAULT_TEAM_COLORS, **(team_colors or {})}
    team_ids = sorted(colors)
    palette_hex = [NEUTRAL, REFEREE, *(colors[t] for t in team_ids)]
    palette = sv.ColorPalette([sv.Color.from_hex(c) for c in palette_hex])
    text_palette = sv.ColorPalette([sv.Color.from_hex(_text_on(c)) for c in palette_hex])
    team_index = {t: i + 2 for i, t in enumerate(team_ids)}
    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    labeler = sv.LabelAnnotator(
        color=palette,
        text_color=text_palette,
        text_scale=0.35,
        text_thickness=1,
        text_padding=3,
        text_position=sv.Position.TOP_CENTER,
    )
    triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex(BALL),
        base=16,
        height=13,
        position=sv.Position.TOP_CENTER,
        color_lookup=sv.ColorLookup.INDEX,
        outline_thickness=1,
    )

    df = detections
    by_frame: dict[int, pd.DataFrame] = {int(f): g for f, g in df.groupby("frame")} if len(df) else {}
    balls = df[df["cls"] == "ball"].sort_values("t_s") if len(df) else df
    ball_t = balls["t_s"].to_numpy(dtype=float) if len(balls) else np.array([])
    ball_xy = (
        np.column_stack([(balls["x1"] + balls["x2"]) / 2, (balls["y1"] + balls["y2"]) / 2])
        if len(balls)
        else np.zeros((0, 2))
    )
    flashes = [
        (lab, ev.t_key_s if ev.t_key_s is not None else ev.t_start_s)
        for ev in events
        if (lab := flash_label(ev))
    ]
    fixed_rim = tuple(float(v) for v in rim_box) if rim_box is not None else None
    rim_rows = df[df["cls"] == "rim"] if len(df) else df
    rim_bottom = fixed_rim[3] if fixed_rim else (float(rim_rows["y2"].median()) if len(rim_rows) else None)
    # The flash goes at the top unless the rim is up there (keep the rim/net readable).
    banner_y = height - 70 if rim_bottom is not None and rim_bottom < 110 else 16

    target = out_path if master_audio is None else out_path.with_name(out_path.stem + ".video.mp4")
    with VideoWriter(target, width, height, rate) as writer:
        for i, (t, rgb) in enumerate(iter_frames(proxy_path)):
            img = np.ascontiguousarray(rgb[..., ::-1])
            g = by_frame.get(i)
            rims: list[tuple[float, ...]] = [fixed_rim] if fixed_rim else []
            if g is not None:
                _draw_people(img, g, ellipse, labeler, team_index)
                if not fixed_rim:
                    rims += [tuple(r) for r in g.loc[g["cls"] == "rim", ["x1", "y1", "x2", "y2"]].to_numpy()]
                gb = g[g["cls"] == "ball"]
                real = gb[~gb["interpolated"].astype(bool)]
                if len(real):
                    triangle.annotate(img, sv.Detections(xyxy=real[["x1", "y1", "x2", "y2"]].to_numpy(float)))
                for r in gb[gb["interpolated"].astype(bool)].itertuples():
                    c = (int((r.x1 + r.x2) / 2), int((r.y1 + r.y2) / 2))
                    cv2.circle(img, c, max(4, int((r.x2 - r.x1) / 2)), _bgr(BALL), 1, cv2.LINE_AA)
            for x1, y1, x2, y2 in rims:
                cv2.rectangle(
                    img, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), _bgr(RIM), 1, cv2.LINE_AA
                )
            _draw_trail(img, ball_t, ball_xy, t)
            _chip(img, _clock(t), (12, 12), "#000000", 0.5, alpha=0.55)
            for label, key_t in flashes:
                dt = abs(t - key_t)
                if dt <= FLASH_S:
                    scale = 1.1
                    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                    x = max(0, (width - tw) // 2 - 16)
                    _chip(
                        img, label, (x, banner_y), FLASH_COLORS[label], scale, alpha=0.9 * (1 - dt / FLASH_S)
                    )
                    break
            writer.write(img[..., ::-1])
            if progress is not None and i % 30 == 0:
                frac = min(i / n_frames, 0.99) if n_frames else 0.0
                progress(frac, f"Drawing annotated video ({i}/{n_frames or '?'} frames)")

    if master_audio is not None:
        mux_audio(target, Path(master_audio), out_path)
        target.unlink(missing_ok=True)
    if progress is not None:
        progress(1.0, "Annotated video ready")
    return out_path


def _draw_people(
    img: np.ndarray, g: pd.DataFrame, ellipse: Any, labeler: Any, team_index: dict[str, int]
) -> None:
    import supervision as sv

    p = g[g["cls"].isin(PEOPLE)]
    if not len(p):
        return
    lookup = np.array(
        [
            1 if c == "referee" else team_index.get(str(tm), 0)
            for c, tm in zip(p["cls"], p["team_id"], strict=True)
        ],
        dtype=int,
    )
    dets = sv.Detections(
        xyxy=p[["x1", "y1", "x2", "y2"]].to_numpy(float), tracker_id=p["track_id"].to_numpy(int)
    )
    ellipse.annotate(img, dets, custom_color_lookup=lookup)
    tracked = p["track_id"].to_numpy(int) >= 0
    if tracked.any():
        labels = [
            f"{'ref' if c == 'referee' else ''}#{tid}"
            for c, tid in zip(p["cls"][tracked], p["track_id"][tracked], strict=True)
        ]
        labeler.annotate(img, dets[tracked], labels=labels, custom_color_lookup=lookup[tracked])


def _draw_trail(img: np.ndarray, ball_t: np.ndarray, ball_xy: np.ndarray, t: float) -> None:
    import cv2

    if not len(ball_t):
        return
    lo, hi = np.searchsorted(ball_t, t - TRAIL_S, "right"), np.searchsorted(ball_t, t + 1e-6, "right")
    if hi - lo < 2:
        return
    pts, ts = ball_xy[lo:hi], ball_t[lo:hi]
    for k in range(1, len(pts)):
        if ts[k] - ts[k - 1] > 0.25:  # don't bridge long gaps
            continue
        age = (t - ts[k]) / TRAIL_S
        thickness = max(1, round(4 * (1 - age)))
        p0, p1 = tuple(int(v) for v in pts[k - 1]), tuple(int(v) for v in pts[k])
        cv2.line(img, p0, p1, _bgr(BALL), thickness, cv2.LINE_AA)
