from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hoops.charts import MADE, MISSED, court_heatmap, image_heatmap, shot_chart


def _png(path):
    img = Image.open(path)
    assert img.format == "PNG"
    return np.asarray(img.convert("RGB"))


def _has_color(rgb, color, tol=12):
    return bool((np.abs(rgb.astype(int) - color).max(axis=2) <= tol).any())


@pytest.mark.parametrize("court", ["nba", "fiba", "hs"])
def test_shot_chart_half_and_full(tmp_path, court):
    shots = [(25.0, 0.0, True), (40.0, 18.0, False), (-30.0, -5.0, True), (44.0, -2.0, False)]
    half = _png(shot_chart(shots, tmp_path / "half.png", court=court, title="Game 1"))
    full = _png(shot_chart(shots, tmp_path / "full.png", court=court, half=False))
    assert full.shape[1] > full.shape[0] and half.shape[1] < full.shape[1]
    for img in (half, full):
        assert _has_color(img, MADE) and _has_color(img, MISSED)


def test_shot_chart_empty(tmp_path):
    out = shot_chart([], tmp_path / "charts" / "empty.png")
    assert _png(out).shape == (676, 656, 3)  # NBA half court at 12 px/ft plus header and footer


def test_court_heatmap(tmp_path):
    rng = np.random.default_rng(1)
    pts = np.r_[rng.normal([35, 5], 3, (300, 2)), rng.normal([-20, -10], 4, (100, 2))]
    full = _png(court_heatmap(pts, tmp_path / "heat.png"))
    half = _png(court_heatmap(pts, tmp_path / "heat_half.png", half=True))
    assert full.shape[1] > full.shape[0] and half.shape[0] > half.shape[1]  # NBA half: taller than wide
    assert court_heatmap(np.zeros((0, 2)), tmp_path / "none.png").exists()


def test_image_heatmap(tmp_path):
    frame = np.full((360, 640, 3), 120, np.uint8)
    pts = np.random.default_rng(2).normal([320, 250], 30, (200, 2))
    img = _png(image_heatmap(pts, frame, tmp_path / "img_heat.png"))
    assert img.shape[1] == 640 and img.shape[0] > 360
    assert image_heatmap(np.zeros((0, 2)), frame, tmp_path / "empty.png").exists()
