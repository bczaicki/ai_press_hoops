"""Numbers -> descriptors for shot candidates.

Code does all the arithmetic here. System One models (Jev can't count, compare numbers, or do math)
only see the short categorical ``descriptors``; the numeric ``features`` are logged per candidate so
thresholds can be recalibrated and re-applied later without re-running inference.

Geometry is rim-relative. For each ball point, ``dx`` is the ball-center offset from the rim center
in rim widths, and ``dy`` is the offset in rim heights. Image y grows downward, so a negative ``dy``
is above the rim. ``dyw`` is the same vertical offset measured in rim widths. Use it for distances
that must not blow up when a side-on view makes the rim box flat.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

RimBox = tuple[float, float, float, float]
PEOPLE = ("player", "person")
EVENT_CLASSES = ("ball_in_basket", "jump_shot", "layup_dunk", "shot_block")

UP, NEAR, DOWN, AWAY = "up", "near", "down", "away"
_POINT_COLS = ["t_s", "frame", "x", "y", "score", "interpolated"]


def normalize_detections(detections: pd.DataFrame) -> pd.DataFrame:
    """Fill optional DETECTION_COLUMNS with defaults and sort by time."""
    df = detections.copy()
    defaults = {"frame": -1, "score": 1.0, "track_id": -1, "team_id": "", "interpolated": False}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df["interpolated"] = df["interpolated"].fillna(False).astype(bool)
    df["track_id"] = df["track_id"].fillna(-1).astype(int)
    df["team_id"] = df["team_id"].fillna("").astype(str)
    return df.sort_values("t_s", kind="stable").reset_index(drop=True)


def frame_interval(detections: pd.DataFrame) -> float:
    """Median spacing of analyzed frames (seconds); 1/30 when unknown."""
    ts = np.unique(detections["t_s"].to_numpy(dtype=float)) if len(detections) else np.array([])
    d = np.diff(ts)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else 1 / 30


def rim_track(detections: pd.DataFrame, *, min_score: float = 0.5) -> pd.DataFrame:
    """Top-1 rim box per analyzed frame (columns t_s, x1, y1, x2, y2, score).

    Prefers detections scoring >= min_score; falls back to any rim detection when none pass.
    Assumes one active basket in view (side baskets in a wide gym view can confuse it).
    """
    rims = detections[detections["cls"] == "rim"]
    strong = rims[rims["score"] >= min_score]
    if not strong.empty:
        rims = strong
    top = rims.sort_values("score", ascending=False, kind="stable").drop_duplicates("t_s")
    cols = ["t_s", "x1", "y1", "x2", "y2", "score"]
    return top.sort_values("t_s")[cols].reset_index(drop=True)


def rim_box_from_detections(detections: pd.DataFrame, *, min_score: float = 0.5) -> RimBox | None:
    """Median rim box over the clip, or None when no rim was detected."""
    track = rim_track(detections, min_score=min_score)
    if track.empty:
        return None
    x1, y1, x2, y2 = (float(track[c].median()) for c in ("x1", "y1", "x2", "y2"))
    return (x1, y1, x2, y2)


def rim_geometry(box: RimBox) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) of a rim box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)


def ball_points(
    detections: pd.DataFrame,
    *,
    rim_box: RimBox | None = None,
    min_score: float = 0.3,
    near_rim_score: float = 0.15,
    near_rim_rims: float = 2.0,
    max_speed_rims_s: float = 40.0,
) -> pd.DataFrame:
    """Top-1 ball per frame with score and motion gating.

    Keeps a ball if score > min_score, or score > near_rim_score within near_rim_rims rim widths of
    the rim (small, blurred balls at the rim matter most). Rows the perception stage interpolated
    are kept as-is. A motion gate drops jumps faster than max_speed_rims_s rim widths per second
    (plus slack), and resets after 3 straight drops so one false positive can't lock it out.
    Returns columns t_s, frame, x, y, score, interpolated, sorted by time.
    """
    balls = detections[detections["cls"] == "ball"]
    if balls.empty:
        return pd.DataFrame(columns=_POINT_COLS)
    balls = balls.assign(x=(balls["x1"] + balls["x2"]) / 2, y=(balls["y1"] + balls["y2"]) / 2)
    keep = balls["interpolated"] | (balls["score"] > min_score)
    rim_w = None
    if rim_box is not None:
        cx, cy, rim_w, _ = rim_geometry(rim_box)
        near = ((balls["x"] - cx).abs() <= near_rim_rims * rim_w) & (
            (balls["y"] - cy).abs() <= near_rim_rims * rim_w
        )
        keep |= near & (balls["score"] > near_rim_score)
    balls = balls[keep]
    if balls.empty:
        return pd.DataFrame(columns=_POINT_COLS)
    balls = balls.sort_values(["t_s", "interpolated", "score"], ascending=[True, True, False], kind="stable")
    balls = balls.drop_duplicates("t_s")
    if rim_w is None:  # a basketball is about half a rim wide
        rim_w = float(((balls["x2"] - balls["x1"]).median() or 10.0) * 1.9)

    kept: list[int] = []
    last: tuple[float, float, float] | None = None
    drops = 0
    for idx, t, x, y in zip(balls.index, balls["t_s"], balls["x"], balls["y"], strict=True):
        if last is not None and drops < 3:
            dt = t - last[0]
            if dt <= 0.5 and np.hypot(x - last[1], y - last[2]) > (max_speed_rims_s * dt + 2.0) * rim_w:
                drops += 1
                continue
        kept.append(idx)
        last, drops = (float(t), float(x), float(y)), 0
    return balls.loc[kept, _POINT_COLS].reset_index(drop=True)


def rim_relative(
    points: pd.DataFrame, rims: pd.DataFrame | RimBox, *, tolerance_s: float = 1.0
) -> pd.DataFrame:
    """Attach rim geometry to each ball point and compute dx (rim widths), dy (rim heights), dyw.

    ``rims`` is either a static box (fixed camera / user click) or a per-frame rim track (moving
    camera); with a track, each point uses the rim detected nearest in time (within tolerance_s),
    falling back to the median rim.
    """
    out = points.copy()
    if out.empty:
        for col in ("rim_cx", "rim_cy", "rim_w", "rim_h", "dx", "dy", "dyw"):
            out[col] = pd.Series(dtype=float)
        return out
    if isinstance(rims, pd.DataFrame) and not rims.empty:
        track = rims.assign(
            rim_cx=(rims["x1"] + rims["x2"]) / 2,
            rim_cy=(rims["y1"] + rims["y2"]) / 2,
            rim_w=(rims["x2"] - rims["x1"]).clip(lower=1e-6),
            rim_h=(rims["y2"] - rims["y1"]).clip(lower=1e-6),
        )[["t_s", "rim_cx", "rim_cy", "rim_w", "rim_h"]]
        out = out.astype({"t_s": float}).sort_values("t_s")
        out = pd.merge_asof(
            out, track.astype({"t_s": float}), on="t_s", direction="nearest", tolerance=tolerance_s
        )
        fallback = track[["rim_cx", "rim_cy", "rim_w", "rim_h"]].median()
        for col in ("rim_cx", "rim_cy", "rim_w", "rim_h"):
            out[col] = out[col].fillna(fallback[col])
    else:
        cx, cy, w, h = rim_geometry(rims)  # type: ignore[arg-type]
        out = out.assign(rim_cx=cx, rim_cy=cy, rim_w=w, rim_h=h)
    out["dx"] = (out["x"] - out["rim_cx"]) / out["rim_w"]
    out["dy"] = (out["y"] - out["rim_cy"]) / out["rim_h"]
    out["dyw"] = (out["y"] - out["rim_cy"]) / out["rim_w"]
    return out.reset_index(drop=True)


def zone(
    dx: float,
    dy: float,
    *,
    up_halfwidth: float = 4.0,
    up_height: float = 2.0,
    down_below: float = 0.5,
    down_halfwidth: float = 6.0,
    near_halfwidth: float = 1.0,
) -> str:
    """Classify a rim-relative ball position.

    - DOWN: more than ``down_below`` rim heights below the rim center (and not far off to the side).
    - UP: within ±``up_halfwidth`` rim widths, between the rim's top edge and ``up_height`` rim
      heights above it.
    - NEAR: on the rim itself (the rim box widened to ±``near_halfwidth`` rim widths).
    - AWAY: anywhere else.
    """
    if dy > down_below and abs(dx) <= down_halfwidth:
        return DOWN
    if -0.5 - up_height <= dy < -0.5 and abs(dx) <= up_halfwidth:
        return UP
    if abs(dx) <= near_halfwidth and -0.5 <= dy <= down_below:
        return NEAR
    return AWAY


def coverage(times: np.ndarray, t0: float, t1: float, frame_dt: float) -> float:
    """Fraction of analyzed frames in [t0, t1] that have an observation."""
    if t1 < t0:
        return 0.0
    expected = max(1, round((t1 - t0) / frame_dt) + 1)
    seen = int(np.count_nonzero((times >= t0 - 1e-9) & (times <= t1 + 1e-9)))
    return float(min(1.0, seen / expected))


def longest_run(times: np.ndarray, frame_dt: float) -> int:
    """Longest run of consecutive analyzed frames among sorted, unique times."""
    if len(times) == 0:
        return 0
    best = run = 1
    for a, b in itertools.pairwise(times):
        run = run + 1 if b - a <= 1.5 * frame_dt else 1
        best = max(best, run)
    return best


def describe(f: dict[str, Any], *, make_halfwidth: float = 0.4) -> dict[str, str]:
    """Qualitative descriptors (short categorical text) from a candidate's numeric features."""
    margin = f.get("crossing_margin")
    bounced = bool(f.get("bounced"))
    if margin is None:
        crossing = "unclear"
        path = "rose above rim, then was lost near the rim"
    elif margin <= make_halfwidth:
        crossing = "inside rim edges"
        path = (
            "bounced on the rim, then dropped through the rim"
            if bounced
            else ("rose above rim, then dropped through rim region")
        )
    elif margin <= 0.6:
        crossing = "at the rim edge"
        path = (
            "bounced on the rim, then fell at the rim edge"
            if bounced
            else ("rose above rim, then came down at the rim edge")
        )
    else:
        crossing = "outside rim edges"
        if bounced:
            path = "bounced off the rim, then fell outside the rim"
        elif margin <= 1.5:
            path = "rose above rim, then fell beside the rim"
        else:
            path = "rose above rim, then fell well away from the rim"
    if f.get("trigger") == "ball_in_basket":
        path = "ball path not tracked; ball-in-basket detector fired"

    cov = float(f.get("ball_coverage") or 0.0)
    visible = "most frames" if cov >= 0.7 else "some frames" if cov >= 0.3 else "few frames"

    if not f.get("event_classes_available", True):
        bib = "not available"
    elif int(f.get("ball_in_basket_run") or 0) >= 3:
        bib = "seen in several consecutive frames"
    elif int(f.get("n_ball_in_basket") or 0) > 0:
        bib = "seen briefly"
    else:
        bib = "not seen"

    if f.get("rim_source") == "user":
        rim = "marked by user"
    else:
        rv = float(f.get("rim_visible_frac") or 0.0)
        rim = "throughout" if rv >= 0.8 else "partially" if rv >= 0.3 else "rarely"

    dist = f.get("shooter_distance_rims")
    if dist is None:
        shooter_distance = "unknown"
    else:
        shooter_distance = (
            "close to the rim" if dist <= 3 else "mid-range" if dist <= 8 else "far from the rim"
        )

    apex = f.get("apex_above_rim_w")
    if apex is None:
        arc = "unknown"
    else:
        arc = "high above the rim" if apex >= 2.0 else "just above the rim"

    return {
        "ball_path": path,
        "crossing": crossing,
        "ball_visible": visible,
        "ball_in_basket_class": bib,
        "rim_visible": rim,
        "scene_cut_nearby": "yes" if f.get("near_scene_cut") else "no",
        "shooter_motion": str(f.get("shooter_motion") or "unknown"),
        "shooter_distance": shooter_distance,
        "arc": arc,
    }
