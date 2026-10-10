"""Synthetic detections for shot tests: ballistic ball arcs at a fixed rim, plus a shooter box."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hoops.schemas import DETECTION_COLUMNS

RIM = (620.0, 300.0, 660.0, 316.0)  # w=40, h=16, center (640, 308)
FPS = 30
G = 1890.0  # px/s^2; with the defaults below the apex is 150 px above the rim


def _row(frame, t, cls, box, score=0.9, track_id=-1, team_id="", interpolated=False):
    x1, y1, x2, y2 = box
    return {
        "frame": frame,
        "t_s": round(t, 6),
        "cls": cls,
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "track_id": track_id,
        "team_id": team_id,
        "interpolated": interpolated,
    }


def ball_box(x, y, r=10.0):
    return (x - r, y - r, x + r, y + r)


def arc(t0, start=(400.0, 500.0), target=(640.0, 308.0), flight=1.0, after="drop", after_s=0.5, fps=FPS):
    """Ball positions for a shot released at t0 that reaches `target` after `flight` seconds.

    after: "drop" (falls straight down through the net), "carry" (keeps its horizontal speed),
    or "bounce" (pops up and away off the rim, then falls).
    """
    (x0, y0), (xt, yt) = start, target
    v = (yt - y0 - 0.5 * G * flight**2) / flight
    vx = (xt - x0) / flight
    pts = []
    n = round((flight + after_s) * fps)
    for k in range(n + 1):
        tau = k / fps
        t = t0 + tau
        if tau <= flight:
            x, y = x0 + vx * tau, y0 + v * tau + 0.5 * G * tau**2
        elif after == "drop":
            s = tau - flight
            x, y = xt, yt + (v + G * flight) * s + 0.5 * G * s**2
        elif after == "carry":
            x, y = x0 + vx * tau, y0 + v * tau + 0.5 * G * tau**2
        else:  # bounce: up and out at 300 px/s horizontally, -500 px/s vertically
            s = tau - flight
            x, y = xt + 300 * s, yt - 500 * s + 0.5 * G * s**2
        pts.append((t, x, y))
    return pts


def detections(
    ball: list[tuple[float, float, float]],
    *,
    duration: float | None = None,
    rim: tuple[float, float, float, float] | None = RIM,
    shooter: tuple[float, float, float, float] | None = (380.0, 420.0, 420.0, 560.0),
    extra: list[dict] | None = None,
    fps: int = FPS,
    drop_ball_every: int = 0,
) -> pd.DataFrame:
    end = duration if duration is not None else (max(t for t, _, _ in ball) + 0.5 if ball else 3.0)
    rows = []
    ball_at = {round(t * fps): (x, y) for t, x, y in ball}
    for f in range(round(end * fps) + 1):
        t = f / fps
        if rim is not None:
            rows.append(_row(f, t, "rim", rim, score=0.8))
        if shooter is not None:
            rows.append(_row(f, t, "player", shooter, score=0.9, track_id=7, team_id="A"))
        if f in ball_at and not (drop_ball_every and f % drop_ball_every == 0):
            rows.append(_row(f, t, "ball", ball_box(*ball_at[f]), score=0.8))
    rows += extra or []
    return pd.DataFrame(rows, columns=list(DETECTION_COLUMNS))


def held(t0, t1, xy=(400.0, 500.0), fps=FPS):
    """Ball held at xy (in the shooter's hands) from t0 to t1."""
    return [(k / fps, xy[0], xy[1]) for k in range(round(t0 * fps), round(t1 * fps))]


def class_rows(cls, t0, t1, box=RIM, fps=FPS, score=0.7):
    return [_row(f, f / fps, cls, box, score=score) for f in range(round(t0 * fps), round(t1 * fps) + 1)]


def dribble(t0, t1, x=640.0, fps=FPS):
    """Ball bouncing between y=450 and y=600 (always below the rim)."""
    return [
        (k / fps, x, 525 + 75 * np.cos(2 * np.pi * 2 * k / fps)) for k in range(int(t0 * fps), int(t1 * fps))
    ]
