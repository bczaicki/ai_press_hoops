from __future__ import annotations

import numpy as np
import pytest

from hoops.frames import crop, frames_at, iter_frames, to_jpeg
from hoops.video_io import VideoWriter, is_browser_playable
from tests.helpers import requires_ffmpeg


@requires_ffmpeg
def test_iter_frames_window_and_fps(testsrc_video):
    frames = list(iter_frames(testsrc_video, t0=0.5, t1=1.5, fps=10))
    assert 9 <= len(frames) <= 11
    times = [t for t, _ in frames]
    assert times == sorted(times) and times[0] >= 0.5 - 1e-6 and times[-1] < 1.5
    assert frames[0][1].shape == (360, 640, 3) and frames[0][1].dtype == np.uint8


@requires_ffmpeg
def test_frames_at_nearest(testsrc_video):
    got = frames_at(testsrc_video, [1.0, 0.2])
    assert [round(t, 1) for t, _ in got] == [1.0, 0.2]


@requires_ffmpeg
@pytest.mark.parametrize("size", [(640, 360), (641, 361)])
def test_writer_is_browser_playable(tmp_path, size):
    w, h = size
    out = tmp_path / "out.mp4"
    with VideoWriter(out, w, h, "30000/1001") as vw:
        for i in range(15):
            vw.write(np.full((h, w, 3), i * 10, dtype=np.uint8))
    assert is_browser_playable(out)


def test_to_jpeg_and_crop():
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert to_jpeg(rgb, max_px=640)[:2] == b"\xff\xd8"
    c = crop(rgb, (600, 300, 620, 320), pad=0.25, min_size=64)
    assert c.shape[0] >= 64 and c.shape[1] >= 64
