from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from hoops.reason.sheets import clock, contact_sheet, mmss, overview_images, resolve_ref, window_images
from tests.helpers import requires_ffmpeg


def _frames(n: int, w: int = 320, h: int = 180) -> list[np.ndarray]:
    return [np.full((h, w, 3), 40 * i % 255, dtype=np.uint8) for i in range(n)]


def test_time_formats():
    assert mmss(72.4) == "01:12.40"
    assert mmss(5) == "00:05.00"
    assert clock(72.9) == "1:12"


def test_contact_sheet_layout_index_and_chip():
    sheet, index = contact_sheet(_frames(5), [0.0, 1.0, 2.5, 3.0, 61.25], "S2", cols=3, tile=(320, 180))
    assert sheet.size == (960, 360)
    assert list(index) == ["S2#1", "S2#2", "S2#3", "S2#4", "S2#5"]
    assert index["S2#5"] == 61.25
    # the label chip is yellow text on black in the tile's top-left corner
    chip = np.asarray(sheet.crop((0, 0, 160, 40)))
    yellow = (chip[..., 0] > 200) & (chip[..., 1] > 180) & (chip[..., 2] < 80)
    assert yellow.sum() > 20


def test_contact_sheet_rejects_bad_input():
    with pytest.raises(ValueError):
        contact_sheet(_frames(2), [0.0], "S1")
    with pytest.raises(ValueError):
        contact_sheet([], [], "S1")


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("S2#4", 12.4),
        ("s2#4", 12.4),
        ("see tile S2#4 (rim)", 12.4),
        ("W1#2 @ 00:03.50", 3.5),  # tile wins when known
        ("01:12.5", 72.5),
        ("1:02:03", 3723.0),
        ("12.4s", 12.4),
        ("7", 7.0),
        ("S9#9", None),
        ("unclear", None),
        ("", None),
    ],
)
def test_resolve_ref(ref, expected):
    index = {"S2#4": 12.4, "W1#2": 3.5}
    assert resolve_ref(ref, index) == expected


@requires_ffmpeg
def test_window_images_wide_and_crops(testsrc_video):
    parts, index = window_images(testsrc_video, 0.5, 1.5, fps=6.0)
    assert len(index) == 6
    assert all(0.5 - 1e-6 <= t < 1.5 for t in index.values())
    assert len(parts) == 2  # 2x2 sheets of 640x360 at max_px=1280
    img = Image.open(io.BytesIO(parts[0].jpeg))
    assert max(img.size) <= 1280
    assert parts[0].label.startswith("Sheet W1: tiles W1#1..W1#4")

    crops, cidx = window_images(
        testsrc_video, 0.0, 1.0, fps=8.0, crop_box=(300, 150, 340, 180), sheet_prefix="R"
    )
    assert len(crops) == 1 and len(cidx) == 8 and "R1#8" in cidx
    assert Image.open(io.BytesIO(crops[0].jpeg)).size == (1280, 640)


@requires_ffmpeg
def test_overview_images(testsrc_video):
    parts, index = overview_images(testsrc_video, 2.0, fps=2.0)
    assert len(parts) == 1 and len(index) == 4
    assert next(iter(index)) == "S1#1"
    img = Image.open(io.BytesIO(parts[0].jpeg))
    assert img.size == (1920, 720)
    # sparse sampling seeks instead of decoding everything
    parts, index = overview_images(testsrc_video, 2.0, fps=0.4)
    assert len(index) == 1
