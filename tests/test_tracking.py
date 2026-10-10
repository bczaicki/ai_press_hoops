from __future__ import annotations

import numpy as np
import pytest

from hoops import registry
from hoops.perception.detectors import dets_from_rows, empty_detections
from hoops.perception.tracking import BallFilter, RimEstimator, court_mask, points_in_polygon

FPS = 30.0


def ball(cx: float, cy: float, score: float = 0.9, r: float = 6.0):
    return ("ball", score, cx - r, cy - r, cx + r, cy + r)


def run_filter(f: BallFilter, per_frame: list[list[tuple]], cuts: tuple[int, ...] = ()) -> list:
    for i, rows in enumerate(per_frame):
        if i in cuts:
            f.reset()
        f.update(i, i / FPS, dets_from_rows(rows))
    return f.finalize()


def test_ball_filter_keeps_top1_and_gates_confidence_near_rim():
    rim = (600.0, 300.0, 640.0, 310.0)
    f = BallFilter((1280, 720), rim_box=rim)
    pts = run_filter(
        f,
        [
            [ball(600, 100, 0.9), ball(610, 100, 0.5)],  # top-1 by score
            [ball(605, 100, 0.2)],  # low confidence, far from the rim: rejected
            [ball(610, 250, 0.2)],  # same confidence, within 2 rim-widths of the rim: accepted
        ],
    )
    by_frame = {p.frame: p for p in pts if not p.interpolated}
    assert by_frame[0].score == pytest.approx(0.9)
    assert by_frame[0].cx == pytest.approx(600)
    assert by_frame[2].score == pytest.approx(0.2)
    assert [p.frame for p in pts if p.interpolated] == [1]  # the rejected frame is filled instead


def test_ball_filter_rejects_teleports_and_elongated_boxes():
    f = BallFilter((1280, 720))
    frames = [[ball(100 + 5 * i, 200)] for i in range(5)]
    frames[2] = [ball(1200, 650, 0.95)]  # 1100 px in 1/30 s: impossible
    frames[3] = [("ball", 0.9, 110, 180, 125, 240)]  # 15x60 box: a limb, not a ball
    pts = run_filter(f, frames)
    real = {p.frame: p for p in pts if not p.interpolated}
    assert sorted(real) == [0, 1, 4]
    assert f.rejected_motion == 1
    filled = [p for p in pts if p.interpolated]
    assert [p.frame for p in filled] == [2, 3]  # the rejected frames are filled from the good neighbours
    assert all(100 < p.cx < 125 for p in filled)


def test_ball_filter_reanchors_after_repeated_rejections():
    f = BallFilter((1280, 720), reanchor_after=3)
    frames = [[ball(100, 100)]] + [[ball(900, 500)] for _ in range(4)]
    pts = run_filter(f, frames)
    real = sorted(p.frame for p in pts if not p.interpolated)
    assert real == [0, 3, 4]  # frames 1-2 rejected, then the persistent new position wins


def test_ball_filter_interpolates_short_gaps_only_within_a_scene():
    f = BallFilter((1280, 720), max_gap=3)
    frames: list[list[tuple]] = [[] for _ in range(16)]
    frames[0] = [ball(100, 100)]
    frames[4] = [ball(140, 100)]  # gap of 3: filled
    frames[9] = [ball(150, 100)]  # gap of 4: left empty
    frames[11] = [ball(160, 100)]
    frames[13] = [ball(170, 100)]  # scene cut at 12: frame 12 is not filled across it
    pts = run_filter(f, frames, cuts=(12,))
    filled = {p.frame: p for p in pts if p.interpolated}
    assert sorted(filled) == [1, 2, 3, 10]
    assert filled[2].cx == pytest.approx(120)
    assert filled[2].score == pytest.approx(0.9)
    assert [p.frame for p in pts] == sorted(p.frame for p in pts)


def test_ball_filter_smoothing_reduces_jitter():
    f = BallFilter((1280, 720), smooth=3)
    ys = [100, 106, 100, 106, 100, 106]
    pts = run_filter(f, [[ball(100 + 4 * i, y)] for i, y in enumerate(ys)])
    inner = [p.cy for p in pts[1:-1]]
    assert max(inner) - min(inner) < 6


def test_rim_estimator_median_and_user_box_wins():
    est = RimEstimator()
    for i, jitter in enumerate([0, 1, -1, 2, 0]):
        est.update(i, i / FPS, dets_from_rows([("rim", 0.8, 600 + jitter, 300, 640 + jitter, 310)]))
    est.update(5, 5 / FPS, dets_from_rows([("rim", 0.3, 0, 0, 10, 10)]))  # low confidence: ignored
    assert est.box == pytest.approx((600, 300, 640, 310))
    rows = est.rows()
    assert len(rows) == 6  # stable rim -> one row per analyzed frame, including the miss
    assert all(r[2] == pytest.approx((600, 300, 640, 310)) for r in rows)

    user = RimEstimator((10, 20, 30, 25))
    user.update(0, 0.0, dets_from_rows([("rim", 0.99, 600, 300, 640, 310)]))
    assert user.box == (10.0, 20.0, 30.0, 25.0)
    assert user.rows() == [(0, 0.0, (10.0, 20.0, 30.0, 25.0), 1.0)]


def test_rim_estimator_unstable_rim_falls_back_to_raw_rows():
    est = RimEstimator()
    for i in range(6):  # camera pans: the rim moves 100 px per frame
        est.update(i, i / FPS, dets_from_rows([("rim", 0.8, 100 + 100 * i, 300, 140 + 100 * i, 310)]))
    assert est.box is None
    rows = est.rows()
    assert [r[2][0] for r in rows] == [100 + 100 * i for i in range(6)]


def test_court_mask_drops_people_outside_but_keeps_ball_and_rim():
    poly = np.array([[100, 300], [1180, 300], [1180, 700], [100, 700]])
    dets = dets_from_rows(
        [
            ("player", 0.9, 500, 400, 540, 500),  # feet inside
            ("person", 0.9, 10, 100, 50, 200),  # crowd: feet outside
            ("ball", 0.9, 20, 20, 30, 30),  # outside, but never masked
            ("rim", 0.9, 600, 100, 640, 110),
        ]
    )
    kept = court_mask(dets, poly)
    assert list(kept.data["class_name"]) == ["player", "ball", "rim"]
    assert court_mask(dets, None) is dets
    assert points_in_polygon(np.array([[640, 500], [0, 0]]), poly).tolist() == [True, False]


@pytest.mark.parametrize("key", ["bytetrack", "botsort", "ocsort"])
def test_tracker_wrappers_assign_stable_ids_and_reset(key):
    tracker = registry.trackers.create(key, frame_rate=FPS)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def step(i: int, t: float):
        rows = [
            ("player", 0.9, 100 + 3 * i, 300, 160 + 3 * i, 450),
            ("referee", 0.9, 800 - 3 * i, 300, 860 - 3 * i, 450),
            ("ball", 0.9, 400, 200, 412, 212),  # never tracked
        ]
        return tracker.update(dets_from_rows(rows), frame, t)

    out = None
    for i in range(10):
        out = step(i, i / FPS)
    assert out is not None and set(out.data["class_name"]) == {"player", "referee"}
    first_ids = set(out.tracker_id.tolist())
    assert len(first_ids) == 2 and min(first_ids) >= 0

    tracker.reset()  # scene cut
    for i in range(10, 20):
        out = step(i, i / FPS)
    second_ids = set(out.tracker_id.tolist()) - {-1}
    assert len(second_ids) == 2
    assert not first_ids & second_ids  # IDs never survive a cut and are never reused

    empty = tracker.update(empty_detections(), frame, 1.0)
    assert len(empty) == 0 and "class_name" in empty.data
