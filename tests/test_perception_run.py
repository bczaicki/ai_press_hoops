from __future__ import annotations

import numpy as np
import pytest

from hoops import registry
from hoops.perception.detectors import ScriptedDetector
from hoops.perception.run import PerceptionCancelled, run_perception
from hoops.schemas import DETECTION_COLUMNS
from tests.helpers import make_video, requires_ffmpeg

pytestmark = requires_ffmpeg

RIM = ("rim", 0.8, 300.0, 100.0, 330.0, 108.0)


def script(frame: int, t_s: float) -> list[tuple]:
    x = 50 + 4 * frame
    rows = [
        ("player", 0.9, x, 180, x + 40, 300),
        ("referee", 0.85, 560 - 2 * frame, 170, 600 - 2 * frame, 290),
        ("person", 0.2, 5, 5, 25, 60),  # below conf: dropped
        RIM,
        ("ball_in_basket", 0.4, 305, 100, 325, 120),  # raw evidence row, kept as is
    ]
    if frame % 10 not in (3, 4):  # short ball gaps get interpolated
        rows.append(("ball", 0.8, 200 + 2 * frame, 150, 212 + 2 * frame, 162))
    return rows


@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "proxy.mp4", duration=2.0, size="640x360", fps="30", audio=False)


def test_run_perception_with_scripted_detector(clip):
    calls: list[tuple[float, str]] = []
    res = run_perception(
        clip,
        detector=ScriptedDetector(script),
        tracker_key="bytetrack",
        progress=lambda p, m: calls.append((p, m)),
    )
    df = res.detections
    assert tuple(df.columns) == DETECTION_COLUMNS
    assert res.frames_processed == 60
    assert set(df["cls"]) == {"player", "referee", "ball", "rim", "ball_in_basket"}
    assert df["frame"].dtype == np.int64 and df["interpolated"].dtype == bool
    assert (df["team_id"] == "").all()

    people = df[df["cls"].isin(["player", "referee"])]
    assert (people["track_id"] >= 0).mean() > 0.9  # only the first frame of each track is unconfirmed
    assert len(res.tracks) == 2
    assert {t.cls for t in res.tracks} == {"player", "referee"}
    assert all(t.n_frames >= 55 for t in res.tracks)

    balls = df[df["cls"] == "ball"]
    assert balls["frame"].is_unique
    assert balls["interpolated"].sum() == 12  # frames 3,4,13,14,...,53,54
    assert (balls["track_id"] == -1).all()

    assert res.rim_box == pytest.approx(RIM[2:])
    assert (df["cls"] == "rim").sum() == 60
    assert res.fps_measured > 0
    assert calls[-1][0] == 1.0 and "Finding players & ball" in calls[-1][1]
    assert all(0 <= p <= 1 for p, _ in calls)


def test_scene_cuts_reset_tracks_and_stride(clip):
    res = run_perception(clip, detector=ScriptedDetector(script), scene_cuts_s=[1.0], stride=2)
    df = res.detections
    assert res.frames_processed == 30
    assert sorted(df["frame"].unique()) == list(range(0, 60, 2))
    players = df[(df["cls"] == "player") & (df["track_id"] >= 0)]
    before = set(players.loc[players["t_s"] < 1.0, "track_id"])
    after = set(players.loc[players["t_s"] >= 1.0, "track_id"])
    assert before and after and not before & after


def test_user_rim_box_wins_and_cancel(clip):
    res = run_perception(clip, detector=ScriptedDetector(script), rim_box=(1, 2, 3, 4))
    assert res.rim_box == (1.0, 2.0, 3.0, 4.0)
    assert set(
        map(tuple, res.detections.loc[res.detections["cls"] == "rim", ["x1", "y1", "x2", "y2"]].values)
    ) == {(1.0, 2.0, 3.0, 4.0)}

    seen = {"n": 0}

    def cancel() -> bool:
        seen["n"] += 1
        return seen["n"] > 5

    with pytest.raises(PerceptionCancelled) as exc:
        run_perception(clip, detector=ScriptedDetector(script), cancel=cancel)
    assert exc.value.frames_processed == 5


def test_registry_keys_and_empty_script(clip):
    det = registry.detectors.create("fake:scripted")
    res = run_perception(clip, detector=det, stride=10)
    assert res.detections.empty and tuple(res.detections.columns) == DETECTION_COLUMNS
    assert res.tracks == [] and res.rim_box is None
    assert any("rim" in w for w in res.warnings)
