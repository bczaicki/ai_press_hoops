from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hoops.perception.run import _frame
from hoops.perception.teams import assign_teams, kmeans2, track_teams
from hoops.video_io import VideoWriter

RED, BLUE, STRIPES = (200, 30, 40), (30, 60, 200), (20, 20, 20)
PLAYERS = {  # track_id -> (class, colour, x)
    0: ("player", RED, 40),
    1: ("player", BLUE, 140),
    2: ("player", RED, 240),
    3: ("player", BLUE, 340),
    4: ("referee", STRIPES, 440),
    5: ("person", (230, 230, 230), 540),  # untracked bystander below stays untouched
}


@pytest.fixture
def jerseys_clip(tmp_path):
    path = tmp_path / "jerseys.mp4"
    rows = []
    with VideoWriter(path, 640, 360, 15) as w:
        for f in range(15):
            img = np.full((360, 640, 3), (150, 120, 90), dtype=np.uint8)  # wooden floor
            for tid, (cls, color, x) in PLAYERS.items():
                img[100:260, x : x + 50] = color
                rows.append((f, f / 15, cls, 0.9, x, 100, x + 50, 260, tid if tid != 5 else -1, "", False))
            w.write(img)
    return path, _frame(rows)


def test_hsv_kmeans_splits_two_teams_and_skips_referee(jerseys_clip):
    path, dets = jerseys_clip
    out, teams = assign_teams(path, dets)
    tt = track_teams(out)
    assert tt[0] == tt[2] and tt[1] == tt[3] and tt[0] != tt[1]
    assert 4 not in tt and 5 not in tt  # referee and untracked rows keep team_id ""
    assert (out.loc[out["cls"] == "referee", "team_id"] == "").all()
    assert [t.team_id for t in teams] == ["A", "B"]
    colors = {t.team_id: t.color_hex for t in teams}
    red_team = tt[0]
    r, _g, b = (int(colors[red_team][i : i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and b < 100
    assert list(out.columns) == list(dets.columns) and len(out) == len(dets)


def test_manual_colors_assign_nearest(jerseys_clip):
    path, dets = jerseys_clip
    out, teams = assign_teams(path, dets, manual_colors=("#1d4ed8", "#dc2626"))
    tt = track_teams(out)
    assert tt[1] == tt[3] == "A" and tt[0] == tt[2] == "B"
    assert [t.color_hex for t in teams] == ["#1d4ed8", "#dc2626"]


def test_no_tracks_returns_input_unchanged(jerseys_clip):
    path, dets = jerseys_clip
    only_ref = dets[dets["cls"] == "referee"]
    out, teams = assign_teams(path, only_ref)
    assert teams == [] and out.equals(only_ref)


def test_kmeans2_is_deterministic():
    x = np.array([[0.0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    labels, _ = kmeans2(x)
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert pd.Series(kmeans2(x)[0]).equals(pd.Series(labels))


@pytest.mark.slow
def test_siglip_team_split(jerseys_clip):
    pytest.importorskip("sports")
    path, dets = jerseys_clip
    try:
        out, _teams = assign_teams(path, dets, method="siglip", max_crops_per_track=8)
    except OSError as exc:  # SigLIP weights download needs network
        pytest.skip(f"SigLIP unavailable: {exc}")
    tt = track_teams(out)
    assert tt[0] == tt[2] and tt[1] == tt[3] and tt[0] != tt[1]
