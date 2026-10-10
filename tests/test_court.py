from __future__ import annotations

import json

import numpy as np
import pytest

from hoops.perception import court as ct
from hoops.perception.court import Calibration, court_lines, landmarks


def _camera(eye=(-4.0, -6.0, 22.0), target=(36.0, 0.0, 0.0), f=900.0, size=(1280, 720)):
    """Pinhole camera above the floor looking at the main basket; returns court-ft -> image-px."""
    eye, target = np.array(eye), np.array(target)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd])

    def project(pts_ft):
        p = np.c_[np.asarray(pts_ft, float), np.zeros(len(pts_ft))]
        cam = (p - eye) @ rot.T
        return np.c_[f * cam[:, 0] / cam[:, 2] + size[0] / 2, f * cam[:, 1] / cam[:, 2] + size[1] / 2]

    return project


NAMES = [
    "baseline_left_corner",
    "baseline_right_corner",
    "lane_baseline_left",
    "ft_line_left",
    "ft_line_right",
    "halfcourt_left",
    "three_top",
]


def _calibrate(names=NAMES, court="nba", noise=0.0, swap=None):
    lm = landmarks(court)
    pts = _camera()(np.array([lm[n] for n in names]))
    pts += np.random.default_rng(0).normal(0, noise, pts.shape) if noise else 0
    names = list(names)
    if swap:
        names = [n.replace("left", "TMP").replace("right", "left").replace("TMP", "right") for n in names]
    return Calibration.from_points([tuple(p) for p in pts], names, court=court), pts


def test_landmarks_geometry():
    lm = landmarks("nba")
    assert lm["rim"] == (41.75, 0.0) and lm["baseline_left_corner"] == (47.0, 25.0)
    assert lm["ft_line_right"] == (28.0, -8.0) and lm["three_top"] == (41.75 - 23.75, 0.0)
    assert lm["other_baseline_left_corner"] == (-47.0, -25.0)
    fiba = landmarks("fiba")
    assert fiba["baseline_left_corner"] == pytest.approx((14 * ct.FT_PER_M, 7.5 * ct.FT_PER_M))
    assert landmarks("hs")["baseline_left_corner"] == (42.0, 25.0)


@pytest.mark.parametrize("court", ["nba", "ncaa", "fiba", "hs"])
def test_homography_round_trip(court):
    cal, pts = _calibrate(court=court)
    assert cal.is_valid and not cal.mirrored and cal.reprojection_error_px < 1.0
    lm = landmarks(court)
    truth = np.array([lm[n] for n in NAMES])
    np.testing.assert_allclose(cal.to_court(pts), truth, atol=0.1)
    np.testing.assert_allclose(cal.to_image(truth), pts, atol=1.0)
    rim_px = _camera()(np.array([lm["rim"]]))[0]
    assert cal.to_court(rim_px) == pytest.approx(lm["rim"], abs=0.1)  # single point in, single point out


def test_noisy_clicks_and_outlier_rejection():
    cal, pts = _calibrate(noise=1.0)
    assert cal.is_valid and cal.reprojection_error_px < 3.0
    pts = pts.copy()
    pts[3] += [80, -60]  # one sloppy click
    cal = Calibration.from_points([tuple(p) for p in pts], NAMES)
    assert not cal.inliers[3] and sum(cal.inliers) == len(NAMES) - 1
    assert cal.is_valid and any("ft_line_left" in w for w in cal.warnings)


def test_swapped_left_right_is_flagged():
    cal, _ = _calibrate(swap=True)
    assert cal.mirrored and not cal.is_valid
    assert any("swapped" in w for w in cal.warnings)


def test_four_points_fit_exactly_with_a_warning():
    cal, _ = _calibrate(names=NAMES[:2] + NAMES[3:5])
    assert cal.is_valid and cal.reprojection_error_px < 1e-3
    assert any("5th" in w for w in cal.warnings)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (["rim", "ft_line_left", "ft_line_right", "baseline_left_corner"], "above the floor"),
        (
            ["baseline_left_corner", "three_baseline_left", "lane_baseline_left", "lane_baseline_right"],
            "one line",
        ),
        (["center", "halfcourt_left", "nope", "rim"], "unknown"),
        (["center", "halfcourt_left", "halfcourt_right"], "at least 4"),
    ],
)
def test_bad_inputs(names, match):
    with pytest.raises(ValueError, match=match):
        Calibration.from_points([(float(i), float(i * i)) for i in range(len(names))], names)


def test_dict_round_trip():
    cal, pts = _calibrate()
    d = json.loads(json.dumps(cal.to_dict()))
    again = Calibration.from_dict(d)
    np.testing.assert_allclose(again.to_court(pts), cal.to_court(pts))
    assert again.is_valid and again.landmark_names == NAMES


def test_numpy_dlt_matches_camera():
    lm = landmarks("nba")
    ft = np.array([lm[n] for n in NAMES])
    img = _camera()(ft)
    h = ct._dlt(ft, img)
    np.testing.assert_allclose(ct._apply(h, ft), img, atol=1e-6)


def test_court_lines_and_polygon():
    full, half = court_lines("nba"), court_lines("nba", half=True)
    assert len(full) > len(half) >= 7 and all(p.ndim == 2 and p.shape[1] == 2 for p in full)
    allpts = np.vstack(full)
    assert allpts[:, 0].min() == -47 and allpts[:, 0].max() == 47 and abs(allpts[:, 1]).max() == 25
    assert np.vstack(half)[:, 0].min() >= -1e-9
    three = half[4]  # lane, FT circle, center circle half, outline order: outline, center, lane, ft, three
    assert np.allclose(three[0], [47, 22]) and np.allclose(three[-1], [47, -22])
    cal, _ = _calibrate()
    assert cal.image_polygon(half=True).shape == (4, 2)
    np.testing.assert_allclose(ct.fold_to_half([[-40, 3], [10, -2]]), [[40, -3], [10, -2]])
