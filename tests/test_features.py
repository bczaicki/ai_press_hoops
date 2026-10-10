from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hoops.events import features as F
from tests.shot_fixtures import RIM, detections, held


def test_zones():
    assert F.zone(0.0, -1.5) == F.UP  # just above the rim
    assert F.zone(3.9, -2.4) == F.UP
    assert F.zone(4.5, -1.5) == F.AWAY  # too far to the side
    assert F.zone(0.0, -3.0) == F.AWAY  # far above (apex), not in the UP band
    assert F.zone(0.0, 0.0) == F.NEAR
    assert F.zone(0.2, 0.6) == F.DOWN
    assert F.zone(7.0, 2.0) == F.AWAY


def test_rim_box_and_relative_geometry():
    df = detections(held(0.0, 1.0))
    box = F.rim_box_from_detections(df)
    assert box == pytest.approx(RIM)
    pts = F.ball_points(df, rim_box=box)
    rel = F.rim_relative(pts, F.rim_track(df))
    assert rel["dx"].iloc[0] == pytest.approx((400 - 640) / 40)
    assert rel["dy"].iloc[0] == pytest.approx((500 - 308) / 16)
    assert rel["dyw"].iloc[0] == pytest.approx((500 - 308) / 40)


def test_ball_points_gating():
    rows = [
        # frame 0: two balls -> keep the higher score
        {"t_s": 0.0, "cls": "ball", "score": 0.5, "x1": 100, "y1": 100, "x2": 110, "y2": 110},
        {"t_s": 0.0, "cls": "ball", "score": 0.9, "x1": 500, "y1": 300, "x2": 510, "y2": 310},
        # frame 1: low score far from the rim -> dropped; low score near the rim -> kept
        {"t_s": 0.1, "cls": "ball", "score": 0.2, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"t_s": 0.2, "cls": "ball", "score": 0.2, "x1": 630, "y1": 300, "x2": 650, "y2": 320},
        # frame 3: teleport (impossible speed) -> dropped by the motion gate
        {"t_s": 0.233, "cls": "ball", "score": 0.9, "x1": 5, "y1": 700, "x2": 15, "y2": 710},
    ]
    df = F.normalize_detections(pd.DataFrame(rows))
    pts = F.ball_points(df, rim_box=RIM)
    assert list(pts["t_s"]) == [0.0, 0.2]
    assert pts["x"].iloc[0] == 505


def test_coverage_and_runs():
    t = np.array([0.0, 0.1, 0.2, 0.5, 0.6])
    assert F.coverage(t, 0.0, 0.9, 0.1) == pytest.approx(0.5)
    assert F.longest_run(t, 0.1) == 3
    assert F.longest_run(np.array([]), 0.1) == 0


@pytest.mark.parametrize(
    ("feats", "expected"),
    [
        (
            {"crossing_margin": 0.1},
            {"crossing": "inside rim edges", "ball_path": "rose above rim, then dropped through rim region"},
        ),
        ({"crossing_margin": 0.5, "bounced": True}, {"crossing": "at the rim edge"}),
        ({"crossing_margin": 1.0}, {"ball_path": "rose above rim, then fell beside the rim"}),
        ({"crossing_margin": None}, {"crossing": "unclear"}),
        ({"ball_coverage": 0.9}, {"ball_visible": "most frames"}),
        ({"ball_coverage": 0.1}, {"ball_visible": "few frames"}),
        (
            {"ball_in_basket_run": 3, "n_ball_in_basket": 3},
            {"ball_in_basket_class": "seen in several consecutive frames"},
        ),
        ({"n_ball_in_basket": 1, "ball_in_basket_run": 1}, {"ball_in_basket_class": "seen briefly"}),
        ({"event_classes_available": False}, {"ball_in_basket_class": "not available"}),
        ({"rim_visible_frac": 0.5}, {"rim_visible": "partially"}),
        ({"near_scene_cut": True}, {"scene_cut_nearby": "yes"}),
        (
            {"shooter_distance_rims": 2.0, "apex_above_rim_w": 0.8},
            {"shooter_distance": "close to the rim", "arc": "just above the rim"},
        ),
        (
            {"shooter_distance_rims": 12.0, "apex_above_rim_w": 3.0},
            {"shooter_distance": "far from the rim", "arc": "high above the rim"},
        ),
    ],
)
def test_describe(feats, expected):
    d = F.describe(feats)
    for k, v in expected.items():
        assert d[k] == v
    # Descriptors are short words for Jev: no digits it would have to compare.
    assert not any(ch.isdigit() for v in d.values() for ch in v)
