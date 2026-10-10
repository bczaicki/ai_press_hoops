"""Shot candidates from ball + rim detections (a rim-zone state machine).

The idea comes from the avishah3 rim-zone tracker. That repo has no license, so this is a clean
reimplementation, not a copy. Ball positions are made rim-relative (see ``features.py``), then
linearly densified across short gaps so a fast ball can't skip a zone at low sampling rates:

- **Attempt:** the ball enters UP (just above the rim), then reaches DOWN (below the rim) within
  ``up_timeout_s`` of last being UP.
- **Make:** a line fitted to the last ball centers above the rim plane and the first ones below it
  crosses the rim plane within ±``make_halfwidth_rims`` rim widths of the rim center.
- Cooldowns stop the same shot from being counted twice (e.g. a rim bounce). Temporal NMS removes
  duplicates, and a candidate near a scene cut is flagged (it might be a replay or a cut-away).

Scores are documented heuristics in [0, 1], not calibrated probabilities:

- ``attempt_score = 0.55 + 0.35*q_cov + 0.10*[jump_shot/layup_dunk class seen]
  + 0.10*[ball_in_basket class seen] - 0.30*[scene cut nearby] - 0.15*[rim seen in < 30% of
  frames]``, where ``q_cov = min(1, ball coverage / 0.7)``. A clean, fully tracked UP -> DOWN
  sequence scores 0.90.
- ``make_traj = sigmoid((make_halfwidth - margin) / 0.08)``, where margin is the crossing distance
  from the rim center in rim widths. It is 0.99 at the center, 0.87 at 0.25, 0.5 at 0.4, and 0.08
  at 0.6. Reliability shrinks it toward 0.5: less ball coverage near the rim, a scene cut nearby
  (x0.5), a moving rim (x0.7), or a rim bounce (x0.8). Seeing the ``ball_in_basket`` class then
  pulls it toward 1: it removes 75% of the remaining miss probability when seen in at least 2
  consecutive frames, 30% when seen once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hoops.events import features as F


@dataclass
class ShotParams:
    up_halfwidth_rims: float = 4.0  # UP zone: ± rim widths around the rim center
    up_height_rims: float = 2.0  # UP zone: rim heights above the rim's top edge
    down_below_rims: float = 0.5  # DOWN: ball center this many rim heights below the rim center
    down_halfwidth_rims: float = 6.0  # DOWN: not further than this to the side
    make_halfwidth_rims: float = 0.4  # make: crossing within ± rim widths of the rim center
    attempt_cooldown_s: float = 1.5
    make_cooldown_s: float = 2.0
    min_ball_points: int = 3
    nms_s: float = 1.0
    up_timeout_s: float = 2.5  # UP -> DOWN must happen within this long of last being UP
    max_gap_s: float = 0.4  # densify between ball points at most this far apart
    densify_step_s: float = 1 / 60
    ball_min_score: float = 0.3
    ball_near_rim_score: float = 0.15
    rim_min_score: float = 0.5
    fit_points: int = 3  # points per side of the rim plane for the crossing fit
    window_before_s: float = 2.0  # candidate window = [t_key - before, t_key + after]
    window_after_s: float = 1.5
    scene_cut_margin_s: float = 1.0
    bib_min_frames: int = 2  # ball_in_basket frames needed for a standalone candidate
    class_min_score: float = 0.3


@dataclass
class Candidate:
    cand_id: str
    t_start_s: float
    t_key_s: float  # rim-plane crossing (or the DOWN moment when the crossing is unclear)
    t_end_s: float
    attempt_score: float
    make_score: float
    features: dict[str, Any]
    descriptors: dict[str, str]
    rim_box: tuple[float, float, float, float]
    shooter_track_id: int | None
    shooter_xy: tuple[float, float] | None  # feet (bottom-center of the box), proxy pixels
    ball_points: list[tuple[float, float, float]]  # (t_s, x, y), observed only
    near_scene_cut: bool
    shooter_team_id: str | None = None
    trigger: str = "rim_zone"  # "rim_zone" | "ball_in_basket"


def _clip(v: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return float(min(hi, max(lo, v)))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _densify(rel: pd.DataFrame, step_s: float, max_gap_s: float) -> list[tuple[float, float, float]]:
    t, dx, dy = (rel[c].to_numpy(dtype=float) for c in ("t_s", "dx", "dy"))
    out: list[tuple[float, float, float]] = []
    for i in range(len(t)):
        out.append((t[i], dx[i], dy[i]))
        if i + 1 < len(t) and 0 < (gap := t[i + 1] - t[i]) <= max_gap_s:
            for k in range(1, int(gap / step_s)):
                a = k * step_s / gap
                out.append(
                    (t[i] + k * step_s, dx[i] + a * (dx[i + 1] - dx[i]), dy[i] + a * (dy[i + 1] - dy[i]))
                )
    return out


def _class_times(df: pd.DataFrame, cls: str, t0: float, t1: float, min_score: float) -> np.ndarray:
    rows = df[(df["cls"] == cls) & (df["t_s"] >= t0) & (df["t_s"] <= t1) & (df["score"] >= min_score)]
    return np.unique(rows["t_s"].to_numpy(dtype=float))


def _crossing(seg: pd.DataFrame, t_down: float, p: ShotParams) -> tuple[float | None, float | None]:
    """(signed crossing offset in rim widths, crossing time) or (None, None) when unclear."""
    above = seg[(seg["dy"] <= 0) & (seg["t_s"] <= t_down)]
    if above.empty:
        return None, None
    t_a = float(above["t_s"].max())
    after = seg[(seg["dy"] > 0) & (seg["t_s"] > t_a)]
    if after.empty or float(after["t_s"].iloc[0]) - t_a > 2 * p.max_gap_s:
        return None, None
    # Fit only points close to the rim plane (within 1.5 rim widths), so the arc's curvature and a
    # straight drop through the net don't bend the line; fall back to the nearest point per side.
    side_a = above.tail(p.fit_points)
    side_a = side_a[side_a["dyw"] >= -1.5] if (side_a["dyw"] >= -1.5).any() else above.tail(1)
    side_b = after.head(p.fit_points)
    side_b = side_b[side_b["dyw"] <= 1.5] if (side_b["dyw"] <= 1.5).any() else after.head(1)
    pts = pd.concat([side_a, side_b])
    dys, dxs = pts["dy"].to_numpy(dtype=float), pts["dx"].to_numpy(dtype=float)
    if np.ptp(dys) > 1e-6:
        _slope, offset = np.polyfit(dys, dxs, 1)
    else:
        offset = float(dxs.mean())
    a, b = above.iloc[-1], after.iloc[0]
    frac = (0 - a["dy"]) / (b["dy"] - a["dy"]) if b["dy"] != a["dy"] else 0.0
    return float(offset), float(a["t_s"] + frac * (b["t_s"] - a["t_s"]))


def _bounced(seg: pd.DataFrame) -> bool:
    """Ball reached the rim, then rose by more than 0.3 rim widths before falling (rim contact)."""
    at_rim = seg[(seg["dx"].abs() <= 1.0) & (seg["dy"] >= -1.5) & (seg["dy"] <= 0.5)]
    if at_rim.empty:
        return False
    t_contact = float(at_rim["t_s"].iloc[0])
    dyw_contact = float(at_rim["dyw"].iloc[0])
    later = seg[seg["t_s"] > t_contact]
    return bool(not later.empty and float(later["dyw"].min()) < dyw_contact - 0.3)


def _shooter(
    df: pd.DataFrame, obs: pd.DataFrame, t_apex: float
) -> tuple[int | None, tuple[float, float] | None, str | None, float | None]:
    """Nearest player/person to the ball ~0.5-1.0 s before the apex: (track_id, feet_xy, team, box_h)."""
    pre = obs[(obs["t_s"] >= t_apex - 1.5) & (obs["t_s"] <= t_apex)]
    if pre.empty:
        return None, None, None, None
    ref = pre.iloc[int(np.argmin(np.abs(pre["t_s"].to_numpy() - (t_apex - 0.75))))]
    people = df[df["cls"].isin(F.PEOPLE)]
    if people.empty:
        return None, None, None, None
    t_near = people["t_s"].to_numpy(dtype=float)
    t_pick = t_near[int(np.argmin(np.abs(t_near - ref["t_s"])))]
    if abs(t_pick - ref["t_s"]) > 0.2:
        return None, None, None, None
    frame = people[people["t_s"] == t_pick]
    bx, by = float(ref["x"]), float(ref["y"])
    gx = np.maximum(0, np.maximum(frame["x1"] - bx, bx - frame["x2"]))
    gy = np.maximum(0, np.maximum(frame["y1"] - by, by - frame["y2"]))
    box_h = (frame["y2"] - frame["y1"]).clip(lower=1.0)
    dist = (np.hypot(gx, gy) / box_h).to_numpy()
    i = int(np.argmin(dist))
    if dist[i] > 0.75:
        return None, None, None, None
    row = frame.iloc[i]
    tid = int(row["track_id"]) if int(row["track_id"]) >= 0 else None
    team = str(row["team_id"]) or None
    return tid, ((float(row["x1"]) + float(row["x2"])) / 2, float(row["y2"])), team, float(box_h.iloc[i])


def _shooter_motion(
    df: pd.DataFrame, tid: int | None, t_apex: float, box_h: float | None, p: ShotParams
) -> str:
    if len(_class_times(df, "jump_shot", t_apex - 1.5, t_apex + 0.3, p.class_min_score)):
        return "jumping"
    if len(_class_times(df, "layup_dunk", t_apex - 1.5, t_apex + 0.5, p.class_min_score)):
        return "driving to the rim"
    if tid is None or not box_h:
        return "unknown"
    trk = df[(df["track_id"] == tid) & (df["t_s"] >= t_apex - 1.5) & (df["t_s"] <= t_apex + 0.3)]
    if len(trk) >= 3 and float(trk["y2"].max() - trk["y2"].min()) > 0.08 * box_h:
        return "jumping"
    return "unknown"


def _build(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    t_first: float,
    t_down: float,
    rim_source: str,
    rim_times: np.ndarray,
    frame_dt: float,
    classes_available: bool,
    scene_cuts: Sequence[float],
    p: ShotParams,
) -> Candidate:
    seg = obs[(obs["t_s"] >= t_first - 0.2) & (obs["t_s"] <= t_down + 0.5)]
    offset, t_cross = _crossing(seg, t_down, p)
    t_key = t_cross if t_cross is not None else t_down
    margin = abs(offset) if offset is not None else None
    bounced = _bounced(seg[seg["t_s"] <= t_down])

    pre = obs[(obs["t_s"] >= t_first - p.window_before_s) & (obs["t_s"] <= t_down)]
    if pre.empty:
        pre = seg
    i_apex = int(np.argmin(pre["dyw"].to_numpy()))
    t_apex, apex_above = float(pre["t_s"].iloc[i_apex]), float(-pre["dyw"].iloc[i_apex])

    observed_t = obs.loc[~obs["interpolated"].astype(bool), "t_s"].to_numpy(dtype=float)
    cov = F.coverage(observed_t, t_first - 0.5, t_down + 0.3, frame_dt)
    cov_rim = F.coverage(observed_t, t_key - 0.5, t_key + 0.5, frame_dt)
    t0, t1 = max(0.0, t_key - p.window_before_s), t_key + p.window_after_s
    rim_vis = 1.0 if rim_source == "user" else F.coverage(rim_times, t0, t1, frame_dt)
    rim_seg = obs[(obs["t_s"] >= t0) & (obs["t_s"] <= t1)]
    rim_jitter = (
        float(
            np.hypot(rim_seg["rim_cx"].std(ddof=0), rim_seg["rim_cy"].std(ddof=0)) / rim_seg["rim_w"].median()
        )
        if len(rim_seg) > 1
        else 0.0
    )
    near_cut = any(t_first - p.scene_cut_margin_s <= c <= t_down + p.scene_cut_margin_s for c in scene_cuts)

    bib = _class_times(df, "ball_in_basket", t_first - 0.2, t_down + 1.0, p.class_min_score)
    bib_run = F.longest_run(bib, frame_dt)
    action = len(_class_times(df, "jump_shot", t_apex - 1.5, t_apex + 0.3, p.class_min_score)) + len(
        _class_times(df, "layup_dunk", t_apex - 1.5, t_down, p.class_min_score)
    )

    q_cov = min(1.0, cov / 0.7)
    attempt = _clip(
        0.55
        + 0.35 * q_cov
        + 0.10 * (action > 0)
        + 0.10 * (len(bib) > 0)
        - 0.30 * near_cut
        - 0.15 * (rim_vis < 0.3)
    )
    make_traj = 0.5 if margin is None else _sigmoid((p.make_halfwidth_rims - margin) / 0.08)
    reliability = min(1.0, cov_rim / 0.7) * (0.5 if near_cut else 1.0) * (0.7 if rim_jitter > 0.15 else 1.0)
    reliability *= 0.8 if bounced else 1.0
    make = 0.5 + (make_traj - 0.5) * reliability
    if bib_run >= 2:
        make = 1 - (1 - make) * 0.25
    elif len(bib):
        make = 1 - (1 - make) * 0.7
    make = _clip(make)

    tid, feet, team, box_h = _shooter(df, obs, t_apex)
    near_key = obs.iloc[int(np.argmin(np.abs(obs["t_s"].to_numpy() - t_key)))]
    cx, cy, w, h = (float(near_key[c]) for c in ("rim_cx", "rim_cy", "rim_w", "rim_h"))
    shooter_dist = None if feet is None else float(np.hypot(feet[0] - cx, feet[1] - cy) / w)

    feats: dict[str, Any] = {
        "trigger": "rim_zone",
        "t_up_s": round(t_first, 3),
        "t_down_s": round(t_down, 3),
        "t_apex_s": round(t_apex, 3),
        "t_cross_s": None if t_cross is None else round(t_cross, 3),
        "crossing_offset_rims": None if offset is None else round(offset, 4),
        "crossing_margin": None if margin is None else round(margin, 4),
        "apex_above_rim_w": round(apex_above, 3),
        "bounced": bounced,
        "ball_coverage": round(cov, 3),
        "ball_coverage_near_rim": round(cov_rim, 3),
        "n_ball_points": len(seg),
        "rim_visible_frac": round(rim_vis, 3),
        "rim_source": rim_source,
        "rim_jitter_rims": round(rim_jitter, 4),
        "near_scene_cut": near_cut,
        "n_ball_in_basket": len(bib),
        "ball_in_basket_run": int(bib_run),
        "n_action_class": int(action),
        "event_classes_available": classes_available,
        "shooter_motion": _shooter_motion(df, tid, t_apex, box_h, p),
        "shooter_distance_rims": None if shooter_dist is None else round(shooter_dist, 2),
        "make_traj": round(make_traj, 4),
        "reliability": round(reliability, 3),
        "attempt_score": round(attempt, 4),
        "make_score": round(make, 4),
    }
    window = obs[(obs["t_s"] >= t0) & (obs["t_s"] <= t1) & ~obs["interpolated"].astype(bool)]
    return Candidate(
        cand_id=f"cand_{round(t_key * 1000):07d}",
        t_start_s=round(t0, 3),
        t_key_s=round(t_key, 3),
        t_end_s=round(t1, 3),
        attempt_score=attempt,
        make_score=make,
        features=feats,
        descriptors=F.describe(feats, make_halfwidth=p.make_halfwidth_rims),
        rim_box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        shooter_track_id=tid,
        shooter_xy=feet,
        ball_points=[
            (float(t), float(x), float(y))
            for t, x, y in zip(window["t_s"], window["x"], window["y"], strict=True)
        ],
        near_scene_cut=near_cut,
        shooter_team_id=team,
    )


def _bib_candidates(
    df: pd.DataFrame,
    existing: list[Candidate],
    rim: tuple[float, float, float, float],
    frame_dt: float,
    scene_cuts: Sequence[float],
    p: ShotParams,
) -> list[Candidate]:
    """Standalone candidates where the ball_in_basket class fired but no rim-zone attempt was found."""
    times = _class_times(df, "ball_in_basket", -math.inf, math.inf, p.class_min_score)
    out: list[Candidate] = []
    runs: list[list[float]] = []
    for t in times:
        if runs and t - runs[-1][-1] <= 1.5 * frame_dt:
            runs[-1].append(float(t))
        else:
            runs.append([float(t)])
    for run in runs:
        t_key = run[0]
        if len(run) < p.bib_min_frames or any(
            abs(c.t_key_s - t_key) <= max(p.nms_s, p.make_cooldown_s) for c in existing
        ):
            continue
        near_cut = any(abs(c - t_key) <= p.scene_cut_margin_s for c in scene_cuts)
        attempt = _clip(0.6 + 0.1 * (len(run) >= 3) - 0.3 * near_cut)
        make = _clip(0.85 + 0.07 * (len(run) >= 3) - 0.2 * near_cut)
        feats = {
            "trigger": "ball_in_basket",
            "n_ball_in_basket": len(run),
            "ball_in_basket_run": len(run),
            "event_classes_available": True,
            "near_scene_cut": near_cut,
            "rim_source": "detected",
            "rim_visible_frac": 1.0,
            "ball_coverage": 0.0,
            "crossing_margin": None,
            "bounced": False,
            "attempt_score": round(attempt, 4),
            "make_score": round(make, 4),
        }
        t0, t1 = max(0.0, t_key - p.window_before_s), t_key + p.window_after_s
        out.append(
            Candidate(
                cand_id=f"cand_{round(t_key * 1000):07d}",
                t_start_s=round(t0, 3),
                t_key_s=round(t_key, 3),
                t_end_s=round(t1, 3),
                attempt_score=attempt,
                make_score=make,
                features=feats,
                descriptors=F.describe(feats, make_halfwidth=p.make_halfwidth_rims),
                rim_box=rim,
                shooter_track_id=None,
                shooter_xy=None,
                ball_points=[],
                near_scene_cut=near_cut,
                trigger="ball_in_basket",
            )
        )
    return out


def _nms(cands: list[Candidate], nms_s: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.attempt_score):
        if all(abs(c.t_key_s - k.t_key_s) > nms_s for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c.t_key_s)


def detect_candidates(
    detections: pd.DataFrame,
    *,
    rim_box: tuple[float, float, float, float] | None = None,
    params: ShotParams | None = None,
    scene_cuts_s: Sequence[float] = (),
    event_classes_available: bool | None = None,
) -> list[Candidate]:
    """Shot candidates from a detections table (hoops.schemas.DETECTION_COLUMNS, proxy pixels).

    ``rim_box`` is a user-marked rim (fixed camera); otherwise each ball point is measured against
    the rim detected nearest in time (or the median rim). ``event_classes_available`` says whether
    the detector can emit ball_in_basket/jump_shot (None: infer from whether any such row exists).
    """
    p = params or ShotParams()
    if detections is None or detections.empty:
        return []
    df = F.normalize_detections(detections)
    frame_dt = F.frame_interval(df)
    classes_available = (
        bool(df["cls"].isin(F.EVENT_CLASSES).any())
        if event_classes_available is None
        else event_classes_available
    )
    track = F.rim_track(df, min_score=p.rim_min_score)
    if rim_box is not None:
        rims: pd.DataFrame | tuple[float, float, float, float] = tuple(map(float, rim_box))  # type: ignore[assignment]
        static, rim_source, rim_times = tuple(map(float, rim_box)), "user", np.array([])
    elif not track.empty:
        rims, static, rim_source = track, F.rim_box_from_detections(df, min_score=p.rim_min_score), "detected"
        rim_times = track["t_s"].to_numpy(dtype=float)
    else:
        return []
    assert static is not None
    pts = F.ball_points(df, rim_box=static, min_score=p.ball_min_score, near_rim_score=p.ball_near_rim_score)
    cands: list[Candidate] = []
    if len(pts) >= p.min_ball_points:
        obs = F.rim_relative(pts, rims)
        zone_kw = {
            "up_halfwidth": p.up_halfwidth_rims,
            "up_height": p.up_height_rims,
            "down_below": p.down_below_rims,
            "down_halfwidth": p.down_halfwidth_rims,
        }
        in_up, t_first, t_last, cooldown_until = False, 0.0, 0.0, -math.inf
        for t, dx, dy in _densify(obs, p.densify_step_s, p.max_gap_s):
            if t < cooldown_until:
                continue
            z = F.zone(dx, dy, **zone_kw)
            if z == F.UP:
                t_first = t if not in_up else t_first
                in_up, t_last = True, t
            elif in_up and t - t_last > p.up_timeout_s:
                in_up = False
            elif in_up and z == F.DOWN:
                c = _build(
                    df,
                    obs,
                    t_first=t_first,
                    t_down=t,
                    rim_source=rim_source,
                    rim_times=rim_times,
                    frame_dt=frame_dt,
                    classes_available=classes_available,
                    scene_cuts=scene_cuts_s,
                    p=p,
                )
                cands.append(c)
                in_up = False
                cooldown_until = t + (p.make_cooldown_s if c.make_score >= 0.5 else p.attempt_cooldown_s)
    if classes_available:
        cands += _bib_candidates(df, cands, static, frame_dt, scene_cuts_s, p)
    return _nms(cands, p.nms_s)
