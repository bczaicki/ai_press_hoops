from __future__ import annotations

import pandas as pd
import pytest

from hoops.events.shots import Candidate, ShotParams, detect_candidates
from hoops.schemas import DETECTION_COLUMNS
from tests.shot_fixtures import RIM, arc, class_rows, detections, dribble, held


def _one(df, **kw) -> Candidate:
    cands = detect_candidates(df, **kw)
    assert len(cands) == 1, [(c.t_key_s, c.features) for c in cands]
    return cands[0]


def test_clean_make():
    c = _one(detections(held(0.0, 1.0) + arc(1.0)))
    assert c.make_score >= 0.85 and c.attempt_score >= 0.85
    assert c.t_key_s == pytest.approx(2.0, abs=0.05)
    assert c.t_start_s == pytest.approx(c.t_key_s - 2.0, abs=1e-3)
    assert c.t_end_s == pytest.approx(c.t_key_s + 1.5, abs=1e-3)
    assert c.descriptors["crossing"] == "inside rim edges"
    assert c.descriptors["ball_path"] == "rose above rim, then dropped through rim region"
    assert c.descriptors["ball_visible"] == "most frames"
    assert c.descriptors["rim_visible"] == "throughout"
    assert c.descriptors["scene_cut_nearby"] == "no"
    assert c.descriptors["ball_in_basket_class"] == "not available"  # detector has no event classes
    assert c.shooter_track_id == 7 and c.shooter_xy == (400.0, 560.0) and c.shooter_team_id == "A"
    assert c.rim_box == pytest.approx(RIM)
    assert c.ball_points and all(c.t_start_s <= t <= c.t_end_s for t, _, _ in c.ball_points)
    assert not c.near_scene_cut
    assert c.cand_id == f"cand_{round(c.t_key_s * 1000):07d}"


def test_miss_off_the_side():
    c = _one(detections(held(0.0, 1.0) + arc(1.0, target=(680.0, 308.0), after="carry")))
    assert c.make_score <= 0.15
    assert c.descriptors["crossing"] == "outside rim edges"


def test_airball_short():
    c = _one(detections(held(0.0, 1.0) + arc(1.0, target=(560.0, 308.0), after="carry", after_s=0.6)))
    assert c.make_score <= 0.15
    assert c.descriptors["ball_path"] == "rose above rim, then fell well away from the rim"


def test_rim_out_bounce():
    c = _one(detections(held(0.0, 1.0) + arc(1.0, target=(656.0, 300.0), after="bounce", after_s=0.9)))
    assert c.features["bounced"] is True
    assert c.make_score < 0.5
    assert c.descriptors["ball_path"].startswith("bounced off the rim")


def test_dribble_near_rim_is_not_a_shot():
    assert detect_candidates(detections(dribble(0.0, 4.0), duration=4.0)) == []


def test_two_shots_and_cooldown():
    # Shot 1 makes at ~2.0 s. A second UP->DOWN at ~3.2 s falls inside the make cooldown (2.0 s).
    # Shot 3 at ~6.0 s is counted.
    ball = held(0.0, 1.0) + arc(1.0) + arc(2.7, start=(560, 360), flight=0.5) + held(4.0, 5.0) + arc(5.0)
    cands = detect_candidates(detections(ball))
    assert [round(c.t_key_s) for c in cands] == [2, 6]
    loose = detect_candidates(
        detections(ball), params=ShotParams(make_cooldown_s=0.5, attempt_cooldown_s=0.5)
    )
    assert len(loose) == 3


def test_scene_cut_flag_lowers_scores():
    df = detections(held(0.0, 1.0) + arc(1.0))
    clean = _one(df)
    cut = _one(df, scene_cuts_s=[2.4])
    assert cut.near_scene_cut and cut.descriptors["scene_cut_nearby"] == "yes"
    assert cut.attempt_score < clean.attempt_score - 0.2
    assert 0.5 < cut.make_score < clean.make_score


def test_ball_in_basket_votes_and_standalone_trigger():
    voted = _one(detections(held(0.0, 1.0) + arc(1.0), extra=class_rows("ball_in_basket", 2.03, 2.13)))
    assert voted.descriptors["ball_in_basket_class"] == "seen in several consecutive frames"
    assert voted.make_score >= 0.95
    # No ball at all: only the detector's ball_in_basket class fires.
    solo = _one(detections([], duration=4.0, extra=class_rows("ball_in_basket", 2.0, 2.1)))
    assert solo.trigger == "ball_in_basket" and solo.make_score >= 0.85
    assert solo.descriptors["ball_path"].startswith("ball path not tracked")


def test_sparse_sampling_still_detects():
    # Detector stride 3 (10 fps): the UP band is crossed between samples; densifying catches it.
    df = detections(held(0.0, 1.0) + arc(1.0))
    strided = _one(df[df["frame"] % 3 == 0])
    assert strided.make_score >= 0.85 and strided.descriptors["ball_visible"] == "most frames"
    # Ball found in only every 3rd analyzed frame: still a make lean, but less certain.
    patchy = _one(df[(df["cls"] != "ball") | (df["frame"] % 3 == 0)])
    assert 0.6 < patchy.make_score < strided.make_score
    assert patchy.descriptors["ball_visible"] == "some frames"


def test_user_rim_box_and_empty_inputs():
    df = detections(held(0.0, 1.0) + arc(1.0), rim=None)
    assert detect_candidates(df) == []  # no rim detections and no user rim
    c = _one(df, rim_box=RIM)
    assert c.descriptors["rim_visible"] == "marked by user"
    assert detect_candidates(pd.DataFrame(columns=list(DETECTION_COLUMNS))) == []
