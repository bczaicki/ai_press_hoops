"""Tracking (T1): people trackers from Roboflow ``trackers`` 2.6, ball top-1 filtering, rim estimation,
and the court mask. Trackers are registered in ``hoops.registry.trackers``.

The ball is never multi-object tracked: BallFilter keeps the top-1 plausible ball per frame and fills short
gaps. The rim is a stable box (user-drawn box wins) rather than a track.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hoops.registry import trackers

if TYPE_CHECKING:
    import supervision as sv

PEOPLE = ("player", "referee", "person")
Box = tuple[float, float, float, float]


def class_names(dets: sv.Detections) -> np.ndarray:
    names = dets.data.get("class_name")
    return np.asarray(names if names is not None else [""] * len(dets), dtype=object)


def select_classes(dets: sv.Detections, classes: Collection[str]) -> sv.Detections:
    if not len(dets):
        return dets
    wanted = set(classes)
    return dets[np.array([n in wanted for n in class_names(dets)], dtype=bool)]


# ---------------------------------------------------------------- people trackers


class PeopleTracker:
    """Wraps a ``trackers`` tracker. Tracks people only (player/referee/person) and returns only them.

    ``reset()`` (call at scene cuts) builds a fresh tracker; IDs keep increasing so an ID never means two
    different people in one run. Unconfirmed detections come back with ``tracker_id == -1``.
    """

    def __init__(self, key: str, make: Callable[[], Any], uses_frame: bool = False):
        self.key = key
        self._make = make
        self._uses_frame = uses_frame
        self._tracker = make()
        self._offset = 0
        self._max_id = -1
        self.resets = 0

    def update(self, dets: sv.Detections, frame_rgb: np.ndarray | None, t_s: float) -> sv.Detections:
        people = select_classes(dets, PEOPLE)
        if self._uses_frame and frame_rgb is not None:
            bgr = np.ascontiguousarray(frame_rgb[..., ::-1])  # trackers' camera-motion compensation wants BGR
            out = self._tracker.update(people, frame=bgr, timestamp=float(t_s))
        else:
            out = self._tracker.update(people, timestamp=float(t_s))
        if not len(out):
            from hoops.perception.detectors import empty_detections

            out = empty_detections()
            out.tracker_id = np.array([], dtype=int)
            return out
        raw = out.tracker_id if out.tracker_id is not None else np.full(len(out), -1)
        ids = np.where(np.asarray(raw, dtype=int) >= 0, np.asarray(raw, dtype=int) + self._offset, -1)
        self._max_id = max(self._max_id, int(ids.max()))
        out.tracker_id = ids
        return out

    def reset(self) -> None:
        self._tracker = self._make()
        self._offset = self._max_id + 1
        self.resets += 1

    def info(self) -> dict[str, Any]:
        try:
            version = importlib.metadata.version("trackers")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {"impl": self.key, "version": f"trackers=={version}", "license": "Apache-2.0"}


# Thresholds tuned down from the SportsMOT defaults (0.7/0.6): RF-DETR people scores on amateur footage are
# often 0.5-0.8 and would otherwise never start a track.
_TRACK_DEFAULTS = {"track_activation_threshold": 0.5, "high_conf_det_threshold": 0.5, "lost_track_buffer": 30}


@trackers.register(
    "bytetrack",
    label="ByteTrack (fixed camera)",
    license="Apache-2.0",
    requires_modules=("trackers",),
    notes="fast; best for tripod or fixed gym cameras",
)
def _bytetrack(frame_rate: float = 30.0, **kwargs: Any) -> PeopleTracker:
    from trackers import ByteTrackTracker

    params = {"frame_rate": frame_rate, **_TRACK_DEFAULTS, **kwargs}
    return PeopleTracker("bytetrack", lambda: ByteTrackTracker(**params))


@trackers.register(
    "botsort",
    label="BoT-SORT (moving camera)",
    license="Apache-2.0",
    requires_modules=("trackers",),
    notes="camera-motion compensation; for broadcast, panning, or handheld footage",
)
def _botsort(frame_rate: float = 30.0, **kwargs: Any) -> PeopleTracker:
    from trackers import BoTSORTTracker

    params = {"frame_rate": frame_rate, **_TRACK_DEFAULTS, **kwargs}
    return PeopleTracker("botsort", lambda: BoTSORTTracker(**params), uses_frame=True)


@trackers.register(
    "ocsort",
    label="OC-SORT",
    license="Apache-2.0",
    requires_modules=("trackers",),
    notes="robust to non-linear motion and short occlusions",
)
def _ocsort(frame_rate: float = 30.0, **kwargs: Any) -> PeopleTracker:
    from trackers import OCSORTTracker

    params = {"frame_rate": frame_rate, "high_conf_det_threshold": 0.5, "lost_track_buffer": 30, **kwargs}
    return PeopleTracker("ocsort", lambda: OCSORTTracker(**params))


# ---------------------------------------------------------------- ball


@dataclass
class BallPoint:
    frame: int
    t_s: float
    cx: float
    cy: float
    w: float
    h: float
    score: float
    interpolated: bool = False

    @property
    def box(self) -> Box:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)


class BallFilter:
    """Top-1 ball per frame with confidence and motion gating, short-gap interpolation, optional smoothing.

    - Confidence: score > ``conf``, or > ``near_rim_conf`` within ``near_rim_widths`` rim-widths of the rim.
    - Shape: boxes more elongated than ``max_aspect`` are rejected (heads/limbs; blur stretches balls a bit).
    - Motion: a candidate farther than ``max_speed`` (frame diagonals per second) times the elapsed time
      (+ two ball sizes of slack) from the last accepted ball is rejected. After ``reanchor_after``
      consecutive rejections, or ``anchor_ttl_s`` without a ball, the anchor is dropped.
    - Gaps of at most ``max_gap`` analyzed frames inside one scene are linearly interpolated
      (``interpolated=True``, score = the lower neighbor score).

    Call ``update`` for every analyzed frame (with or without a ball), ``reset`` at scene cuts, then
    ``finalize``.
    """

    def __init__(
        self,
        frame_wh: tuple[int, int] = (1280, 720),
        *,
        conf: float = 0.3,
        near_rim_conf: float = 0.15,
        near_rim_widths: float = 2.0,
        rim_box: Box | None = None,
        max_speed: float = 2.5,
        max_aspect: float = 1.8,
        max_gap: int = 8,
        smooth: int = 0,
        reanchor_after: int = 3,
        anchor_ttl_s: float = 0.5,
    ):
        self.diag = math.hypot(*frame_wh)
        self.conf, self.near_rim_conf, self.near_rim_widths = conf, near_rim_conf, near_rim_widths
        self.rim_box = rim_box
        self.max_speed, self.max_aspect, self.max_gap, self.smooth = max_speed, max_aspect, max_gap, smooth
        self.reanchor_after, self.anchor_ttl_s = reanchor_after, anchor_ttl_s
        self._frames: list[tuple[int, float, int]] = []  # every analyzed frame: (frame, t_s, scene segment)
        self._points: dict[int, BallPoint] = {}
        self._segment = 0
        self._last: BallPoint | None = None
        self._rejects = 0
        self.rejected_motion = 0

    def near_rim(self, cx: float, cy: float) -> bool:
        if self.rim_box is None:
            return False
        x1, y1, x2, y2 = self.rim_box
        m = self.near_rim_widths * max(x2 - x1, 1.0)
        return x1 - m <= cx <= x2 + m and y1 - m <= cy <= y2 + m

    def _candidates(self, frame: int, t_s: float, dets: sv.Detections) -> list[BallPoint]:
        balls = select_classes(dets, ("ball",))
        if not len(balls):
            return []
        scores = balls.confidence if balls.confidence is not None else np.ones(len(balls))
        out = []
        for (x1, y1, x2, y2), s in zip(balls.xyxy, scores, strict=True):
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 0 or h <= 0 or max(w / h, h / w) > self.max_aspect:
                continue
            cx, cy = float(x1 + x2) / 2, float(y1 + y2) / 2
            if s > (self.near_rim_conf if self.near_rim(cx, cy) else self.conf):
                out.append(BallPoint(frame, t_s, cx, cy, w, h, float(s)))
        return out

    def _plausible(self, c: BallPoint, last: BallPoint) -> bool:
        dt = max(c.t_s - last.t_s, 1e-3)
        slack = 2.0 * max(c.w, c.h, last.w, last.h)
        return math.hypot(c.cx - last.cx, c.cy - last.cy) <= self.max_speed * self.diag * dt + slack

    def update(self, frame: int, t_s: float, dets: sv.Detections) -> BallPoint | None:
        self._frames.append((frame, t_s, self._segment))
        cands = self._candidates(frame, t_s, dets)
        if self._last is not None and t_s - self._last.t_s > self.anchor_ttl_s:
            self._last = None
        if self._last is not None and cands:
            plausible = [c for c in cands if self._plausible(c, self._last)]
            if not plausible:
                self._rejects += 1
                self.rejected_motion += 1
                if self._rejects < self.reanchor_after:
                    return None
                self._last = None  # the old anchor was probably wrong: start over
                plausible = cands
            cands = plausible
        if not cands:
            return None
        best = max(cands, key=lambda c: c.score)
        self._points[frame] = best
        self._last, self._rejects = best, 0
        return best

    def reset(self) -> None:
        self._segment += 1
        self._last, self._rejects = None, 0

    def _smoothed(self, pts: list[BallPoint], index: dict[int, int]) -> list[BallPoint]:
        k = self.smooth // 2
        out = []
        for i, p in enumerate(pts):
            seg = self._frames[index[p.frame]][2]
            near = [
                q
                for q in pts[max(0, i - k) : i + k + 1]
                if abs(index[q.frame] - index[p.frame]) <= k and self._frames[index[q.frame]][2] == seg
            ]
            cx = sum(q.cx for q in near) / len(near)
            cy = sum(q.cy for q in near) / len(near)
            out.append(BallPoint(p.frame, p.t_s, cx, cy, p.w, p.h, p.score))
        return out

    def finalize(self) -> list[BallPoint]:
        """Accepted balls plus interpolated fills, sorted by frame."""
        index = {f: i for i, (f, _, _) in enumerate(self._frames)}
        pts = sorted(self._points.values(), key=lambda p: p.frame)
        if self.smooth >= 3:
            pts = self._smoothed(pts, index)
        out: list[BallPoint] = []
        for a, b in itertools.pairwise(pts):
            out.append(a)
            ia, ib = index[a.frame], index[b.frame]
            gap = ib - ia - 1
            if not 0 < gap <= self.max_gap or self._frames[ia][2] != self._frames[ib][2]:
                continue
            for k in range(ia + 1, ib):
                f, t, _ = self._frames[k]
                r = (t - a.t_s) / (b.t_s - a.t_s) if b.t_s > a.t_s else (k - ia) / (gap + 1)
                out.append(
                    BallPoint(
                        f,
                        t,
                        a.cx + r * (b.cx - a.cx),
                        a.cy + r * (b.cy - a.cy),
                        a.w + r * (b.w - a.w),
                        a.h + r * (b.h - a.h),
                        min(a.score, b.score),
                        interpolated=True,
                    )
                )
        if pts:
            out.append(pts[-1])
        return out


# ---------------------------------------------------------------- rim


class RimEstimator:
    """A stable rim box. A user-drawn box always wins.

    Otherwise, per scene segment: the median of the top confident rim box per frame (score >= ``min_conf``)
    when those boxes agree (median center jitter <= ``max_jitter`` rim-widths). When they don't (moving
    camera, two hoops in view), ``rows()`` falls back to the raw per-frame rim detections.
    """

    def __init__(
        self,
        user_box: Sequence[float] | None = None,
        *,
        min_conf: float = 0.5,
        min_samples: int = 3,
        max_jitter: float = 0.5,
    ):
        self.user_box: Box | None = (
            tuple(float(v) for v in user_box) if user_box is not None else None  # type: ignore[assignment]
        )
        self.min_conf, self.min_samples, self.max_jitter = min_conf, min_samples, max_jitter
        self._frames: list[tuple[int, float, int]] = []
        self._dets: dict[int, list[tuple[Box, float]]] = {}
        self._segment = 0
        self.latest: Box | None = self.user_box  # most recent confident rim (cheap, for ball gating)

    def update(self, frame: int, t_s: float, dets: sv.Detections) -> None:
        self._frames.append((frame, t_s, self._segment))
        rims = select_classes(dets, ("rim",))
        if not len(rims):
            return
        scores = rims.confidence if rims.confidence is not None else np.ones(len(rims))
        keep = [
            (tuple(float(v) for v in b), float(s))
            for b, s in zip(rims.xyxy, scores, strict=True)
            if s >= self.min_conf
        ]
        if keep:
            self._dets[frame] = keep  # type: ignore[assignment]
            if self.user_box is None:
                self.latest = max(keep, key=lambda d: d[1])[0]  # type: ignore[assignment]

    def reset(self) -> None:
        self._segment += 1
        if self.user_box is None:
            self.latest = None

    def _segments(self) -> list[int]:
        return sorted({s for _, _, s in self._frames})

    def segment_box(self, segment: int) -> tuple[Box, float] | None:
        tops = [
            max(self._dets[f], key=lambda d: d[1])
            for f, _, s in self._frames
            if s == segment and f in self._dets
        ]
        if len(tops) < self.min_samples:
            return None
        boxes = np.array([b for b, _ in tops], dtype=float)
        med = np.median(boxes, axis=0)
        width = max(float(med[2] - med[0]), 1.0)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        jitter = float(np.median(np.hypot(*(centers - (med[:2] + med[2:]) / 2).T)))
        if jitter > self.max_jitter * width:
            return None
        return tuple(float(v) for v in med), float(np.median([s for _, s in tops]))  # type: ignore[return-value]

    @property
    def box(self) -> Box | None:
        """The user box, else the stable box of the longest scene segment that has one."""
        if self.user_box is not None:
            return self.user_box
        best: tuple[int, Box] | None = None
        for seg in self._segments():
            found = self.segment_box(seg)
            n = sum(1 for _, _, s in self._frames if s == seg)
            if found and (best is None or n > best[0]):
                best = (n, found[0])
        return best[1] if best else None

    def rows(self) -> list[tuple[int, float, Box, float]]:
        """Rim rows ``(frame, t_s, box, score)`` for every analyzed frame that has a rim."""
        if self.user_box is not None:
            return [(f, t, self.user_box, 1.0) for f, t, _ in self._frames]
        out: list[tuple[int, float, Box, float]] = []
        for seg in self._segments():
            stable = self.segment_box(seg)
            for f, t, s in self._frames:
                if s != seg:
                    continue
                if stable:
                    out.append((f, t, stable[0], stable[1]))
                else:
                    out.extend((f, t, b, sc) for b, sc in self._dets.get(f, []))
        return out


# ---------------------------------------------------------------- court mask


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorized: points (N, 2), polygon (M, 2) -> bool (N,)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    x, y = pts[:, :1], pts[:, 1:]
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    dy = np.where(qy == py, 1e-12, qy - py)
    crosses = ((py > y) != (qy > y)) & (x < (qx - px) * (y - py) / dy + px)
    return crosses.sum(axis=1) % 2 == 1


def court_mask(
    dets: sv.Detections, polygon: np.ndarray | None, classes: Collection[str] = PEOPLE
) -> sv.Detections:
    """Drop people whose bottom-center (feet) is outside the court polygon (crowd, bench).

    Ball and rim are never masked: both are legitimately above the court in image space.
    """
    if polygon is None or not len(dets):
        return dets
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return dets
    feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
    inside = points_in_polygon(feet, poly)
    targeted = np.array([n in set(classes) for n in class_names(dets)], dtype=bool)
    return dets[inside | ~targeted]
