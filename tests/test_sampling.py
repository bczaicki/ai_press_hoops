from __future__ import annotations

import itertools

import pytest

from hoops.sampling import event_windows, motion_peaks, scene_cuts, uniform_times, window_times
from tests.helpers import make_video, requires_ffmpeg

TWO_SHOTS = (
    "testsrc2=size=320x180:rate=30:duration=2[a];smptebars=size=320x180:rate=30:duration=2[b];"
    "[a][b]concat=n=2:v=1:a=0[out0]"
)
STILL_THEN_MOTION = (
    "color=c=gray:size=320x180:rate=30:duration=2[a];testsrc2=size=320x180:rate=30:duration=2[b];"
    "[a][b]concat=n=2:v=1:a=0[out0]"
)


def test_uniform_times():
    assert uniform_times(2.0, 1) == [0.0, 1.0]
    assert uniform_times(2.05, 1) == [0.0, 1.0, 2.0]
    assert uniform_times(1.0, 4, t0=0.5) == [0.5, 0.75]
    capped = uniform_times(100.0, 2, max_n=4)
    assert capped == [0.0, 25.0, 50.0, 75.0]
    assert uniform_times(0, 2) == [] and uniform_times(5, 0) == []


def test_window_times_half_open():
    ts = window_times(10.0, 13.5, 4)
    assert len(ts) == 14 and ts[0] == 10.0 and ts[-1] == 13.25
    assert window_times(5.0, 5.0, 4) == []


def test_event_windows_merge_and_clamp():
    assert event_windows([1.0, 2.0, 10.0]) == [(0.0, 3.5), (8.0, 11.5)]
    assert event_windows([5.0, 9.1], merge_gap=0.5) == [(3.0, 6.5), (7.1, 10.6)]
    assert event_windows([5.0, 8.6], merge_gap=0.5) == [(3.0, 10.1)]
    assert event_windows([11.0], duration_s=12.0) == [(9.0, 12.0)]
    assert event_windows([]) == []


@requires_ffmpeg
def test_scene_cuts_finds_hard_cut(tmp_path):
    cut = make_video(tmp_path / "cut.mp4", source=TWO_SHOTS, audio=False)
    cuts = scene_cuts(cut)
    assert len(cuts) == 1 and abs(cuts[0] - 2.0) < 0.1
    assert scene_cuts(make_video(tmp_path / "flat.mp4", audio=False)) == []


@requires_ffmpeg
def test_motion_peaks_prefer_moving_part(tmp_path):
    clip = make_video(tmp_path / "still_then_motion.mp4", source=STILL_THEN_MOTION, audio=False)
    peaks = motion_peaks(clip, sample_fps=10, top_k=3, min_gap_s=0.5)
    assert 1 <= len(peaks) <= 3 and peaks == sorted(peaks)
    assert all(t >= 1.9 for t in peaks)
    assert all(b - a >= 0.5 - 1e-6 for a, b in itertools.pairwise(peaks))


@pytest.mark.parametrize("n", [1, 5])
def test_uniform_times_max_n(n):
    assert len(uniform_times(60.0, 1, max_n=n)) == n
