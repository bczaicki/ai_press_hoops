"""T1 perception loop: detect -> court mask -> track people -> ball top-1 -> rim box -> detections table.

The lead's pipeline calls ``run_perception`` and saves ``PerceptionResult.detections`` as
``detections.parquet``. Nothing is written to disk here.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from hoops import registry
from hoops.frames import iter_frames
from hoops.perception.tracking import (
    PEOPLE,
    BallFilter,
    RimEstimator,
    class_names,
    court_mask,
    select_classes,
)
from hoops.schemas import DETECTION_COLUMNS, Track

Box = tuple[float, float, float, float]


class PerceptionCancelled(Exception):
    """Raised when ``cancel()`` returns True; carries how far the loop got."""

    def __init__(self, frames_processed: int):
        super().__init__(f"perception cancelled after {frames_processed} frames")
        self.frames_processed = frames_processed


@dataclass
class PerceptionResult:
    detections: pd.DataFrame  # columns exactly DETECTION_COLUMNS
    tracks: list[Track]
    rim_box: Box | None
    frames_processed: int
    fps_measured: float  # analyzed frames per wall-clock second (decode + detect + track)
    warnings: list[str] = field(default_factory=list)
    detector_info: dict[str, Any] = field(default_factory=dict)
    tracker_info: dict[str, Any] = field(default_factory=dict)


def empty_detections_frame() -> pd.DataFrame:
    return _frame([])


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(DETECTION_COLUMNS))
    return df.astype(
        {
            "frame": "int64",
            "t_s": "float64",
            "cls": "object",
            "score": "float64",
            "x1": "float64",
            "y1": "float64",
            "x2": "float64",
            "y2": "float64",
            "track_id": "int64",
            "team_id": "object",
            "interpolated": "bool",
        }
    )


def _video_meta(path: Path) -> tuple[float, int, int, int]:
    """(fps, n_frames estimate, width, height) of the proxy."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30)
        n = s.frames
        if not n:
            if s.duration and s.time_base:
                duration = float(s.duration * s.time_base)
            else:
                duration = c.duration / av.time_base if c.duration else 0.0
            n = round(duration * fps)
        return fps, int(n), s.codec_context.width, s.codec_context.height


def build_tracks(df: pd.DataFrame) -> list[Track]:
    people = df[(df["track_id"] >= 0) & df["cls"].isin(PEOPLE)]
    tracks = []
    for tid, g in people.groupby("track_id", sort=True):
        tracks.append(
            Track(
                track_id=int(tid),
                cls=str(g["cls"].mode().iloc[0]),
                first_t_s=float(g["t_s"].min()),
                last_t_s=float(g["t_s"].max()),
                n_frames=int(g["frame"].nunique()),
                mean_score=round(float(g["score"].mean()), 4),
                team_id=None,
            )
        )
    return tracks


def run_perception(
    proxy_path: Path,
    *,
    detector_key: str = "rfdetr:coco",
    tracker_key: str = "bytetrack",
    stride: int = 1,
    conf: float = 0.35,
    scene_cuts_s: Sequence[float] = (),
    rim_box: Box | None = None,
    court_polygon: np.ndarray | None = None,
    detector: Any = None,
    tracker: Any = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PerceptionResult:
    """Run detection + tracking over the proxy. Raises ``PerceptionCancelled`` when ``cancel()`` is True."""
    proxy_path = Path(proxy_path)
    stride = max(1, int(stride))
    fps, n_total, width, height = _video_meta(proxy_path)
    if detector is None:
        importlib.import_module("hoops.perception.detectors")  # registers the detectors
        detector = registry.detectors.create(detector_key, conf=conf)
    if tracker is None:
        tracker = registry.trackers.create(tracker_key, frame_rate=fps / stride)
    warnings: list[str] = []
    ball = BallFilter((width, height), rim_box=rim_box)
    rim = RimEstimator(rim_box)
    cuts = sorted(float(c) for c in scene_cuts_s if c > 0)
    cut_i = 0
    rows: list[tuple] = []
    n_analyze = max(1, -(-n_total // stride)) if n_total else 0
    processed = 0
    t_start = time.perf_counter()

    for frame_i, (t, rgb) in enumerate(iter_frames(proxy_path)):
        if frame_i % stride:
            continue
        if cancel is not None and cancel():
            raise PerceptionCancelled(processed)
        crossed = False
        while cut_i < len(cuts) and t >= cuts[cut_i] - 1e-6:
            cut_i, crossed = cut_i + 1, True
        if crossed:  # IDs never survive a cut; the ball can't fly across one either
            tracker.reset()
            ball.reset()
            rim.reset()

        if hasattr(detector, "set_clock"):
            detector.set_clock(frame_i, t)
        dets = court_mask(detector.detect(rgb), court_polygon)

        people = select_classes(dets, PEOPLE)
        if len(people) and people.confidence is not None:
            people = people[people.confidence >= conf]
        tracked = tracker.update(people, rgb, t)
        ids = tracked.tracker_id if tracked.tracker_id is not None else np.full(len(tracked), -1)
        scores = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        for name, s, box, tid in zip(class_names(tracked), scores, tracked.xyxy, ids, strict=True):
            rows.append((frame_i, t, str(name), float(s), *map(float, box), int(tid), "", False))

        rim.update(frame_i, t, dets)
        if rim_box is None:
            ball.rim_box = rim.latest  # lets low-confidence balls near the rim through
        ball.update(frame_i, t, dets)

        is_other = np.array([n not in (*PEOPLE, "ball", "rim") for n in class_names(dets)], dtype=bool)
        others = dets[is_other] if len(dets) else dets
        o_scores = others.confidence if others.confidence is not None else np.ones(len(others))
        for name, s, box in zip(class_names(others), o_scores, others.xyxy, strict=True):
            rows.append((frame_i, t, str(name), float(s), *map(float, box), -1, "", False))

        processed += 1
        if progress is not None and (processed == 1 or processed % 10 == 0):
            done = min(processed / n_analyze, 0.999) if n_analyze else 0.0
            total = f"/{n_analyze}" if n_analyze else ""
            progress(done, f"Finding players & ball ({processed}{total} frames)")

    elapsed = time.perf_counter() - t_start
    for p in ball.finalize():
        rows.append((p.frame, p.t_s, "ball", p.score, *p.box, -1, "", p.interpolated))
    for f, t, box, s in rim.rows():
        rows.append((f, t, "rim", s, *box, -1, "", False))

    df = _frame(rows).sort_values(["frame", "cls", "track_id"], kind="stable").reset_index(drop=True)
    tracks = build_tracks(df)
    rim_final = rim.box

    if processed:
        ball_frames = df.loc[(df["cls"] == "ball") & ~df["interpolated"], "frame"].nunique()
        if ball_frames / processed < 0.2:
            warnings.append(
                f"Ball found in only {ball_frames}/{processed} analyzed frames: shot detection will be unreliable "
                "(try the high-res ball pass or a closer camera)."
            )
        if rim_final is None:
            warnings.append("No stable rim box: draw the rim once (COCO models have no rim class).")
        if not tracks:
            warnings.append("No player tracks were confirmed.")
    if progress is not None:
        progress(1.0, f"Finding players & ball ({processed} frames, {len(tracks)} tracks)")

    return PerceptionResult(
        detections=df,
        tracks=tracks,
        rim_box=rim_final,
        frames_processed=processed,
        fps_measured=round(processed / elapsed, 2) if elapsed > 0 else 0.0,
        warnings=warnings,
        detector_info=_info(detector),
        tracker_info=_info(tracker),
    )


def _info(obj: Any) -> dict[str, Any]:
    try:
        return dict(obj.info()) if hasattr(obj, "info") else {"impl": getattr(obj, "key", type(obj).__name__)}
    except Exception as exc:  # noqa: BLE001 - manifest info must never fail a run
        return {"impl": getattr(obj, "key", type(obj).__name__), "error": str(exc)}
