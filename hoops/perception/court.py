"""Court geometry (feet) and image <-> court calibration from clicked landmarks.

Court frame: origin at center court; x runs along the length toward the "main" basket (the half you
film); y runs across the width, positive to the LEFT of someone at half court facing that basket
(right-handed, z up). Landmark names describe the main half as seen from a camera at half court looking
at the basket. The other half uses the same names with an ``other_`` prefix (the court rotated 180°).

Calibration is a floor-plane homography: only floor landmarks work. The rim is 10 ft up, so clicking it
would skew the fit; ``rim`` in LANDMARKS is the floor point under the rim center (for drawing/distances).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FT_PER_M = 3.28084
RANSAC_PX = 8.0  # inlier threshold, proxy pixels
MAX_REPROJ_PX = 6.0  # RMS error above this means the clicks don't agree with each other


@dataclass(frozen=True)
class CourtSpec:
    key: str
    label: str
    length_ft: float
    width_ft: float
    rim_from_baseline_ft: float  # rim center
    three_arc_ft: float  # arc radius from the rim center
    three_corner_ft: float  # straight corner lines, distance from the long axis
    lane_width_ft: float
    ft_line_from_baseline_ft: float
    ft_circle_r_ft: float = 6.0
    center_circle_r_ft: float = 6.0
    restricted_r_ft: float | None = 4.0
    backboard_from_baseline_ft: float = 4.0
    backboard_width_ft: float = 6.0
    rim_r_ft: float = 0.75

    @property
    def rim_x(self) -> float:
        return self.length_ft / 2 - self.rim_from_baseline_ft


_M = FT_PER_M
COURTS: dict[str, CourtSpec] = {
    "nba": CourtSpec("nba", "NBA", 94, 50, 5.25, 23.75, 22.0, 16.0, 19.0),
    "ncaa": CourtSpec("ncaa", "NCAA", 94, 50, 5.25, 22.146, 21.65, 12.0, 19.0),
    "fiba": CourtSpec(
        "fiba",
        "FIBA",
        28 * _M,
        15 * _M,
        1.575 * _M,
        6.75 * _M,
        6.6 * _M,
        4.9 * _M,
        5.8 * _M,
        ft_circle_r_ft=1.8 * _M,
        center_circle_r_ft=1.8 * _M,
        restricted_r_ft=1.25 * _M,
        backboard_from_baseline_ft=1.2 * _M,
        backboard_width_ft=1.8 * _M,
        rim_r_ft=0.225 * _M,
    ),
    "hs": CourtSpec("hs", "High school (NFHS)", 84, 50, 5.25, 19.75, 19.75, 12.0, 19.0, restricted_r_ft=None),
}
ELEVATED = frozenset({"rim", "other_rim"})


def spec(court: str = "nba") -> CourtSpec:
    try:
        return COURTS[court]
    except KeyError:
        raise ValueError(f"unknown court '{court}'. Known: {sorted(COURTS)}") from None


def landmarks(court: str = "nba") -> dict[str, tuple[float, float]]:
    """Named court points in feet (see the module docstring for the frame and naming)."""
    c = spec(court)
    half_l, half_w, lane = c.length_ft / 2, c.width_ft / 2, c.lane_width_ft / 2
    ft_x = half_l - c.ft_line_from_baseline_ft
    main = {
        "baseline_left_corner": (half_l, half_w),
        "baseline_right_corner": (half_l, -half_w),
        "lane_baseline_left": (half_l, lane),
        "lane_baseline_right": (half_l, -lane),
        "ft_line_left": (ft_x, lane),
        "ft_line_right": (ft_x, -lane),
        "ft_line_center": (ft_x, 0.0),
        "three_baseline_left": (half_l, c.three_corner_ft),
        "three_baseline_right": (half_l, -c.three_corner_ft),
        "three_top": (c.rim_x - c.three_arc_ft, 0.0),
        "rim": (c.rim_x, 0.0),
    }
    shared = {"center": (0.0, 0.0), "halfcourt_left": (0.0, half_w), "halfcourt_right": (0.0, -half_w)}
    other = {f"other_{k}": (-x, -y) for k, (x, y) in main.items()}
    return {**shared, **main, **other}


# ---------------------------------------------------------------- drawing geometry


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 48) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def _half_lines(c: CourtSpec) -> list[np.ndarray]:
    """Markings of the main half (x >= 0), excluding the outline and center circle."""
    half_l, lane, rx = c.length_ft / 2, c.lane_width_ft / 2, c.rim_x
    ft_x = half_l - c.ft_line_from_baseline_ft
    r, corner = c.three_arc_ft, min(c.three_corner_ft, c.three_arc_ft)
    phi0 = math.asin(corner / r)
    phis = np.linspace(phi0, -phi0, 64)  # angle from the -x direction, left corner -> top -> right corner
    arc = np.c_[rx - r * np.cos(phis), r * np.sin(phis)]
    bb_x, bb_w = half_l - c.backboard_from_baseline_ft, c.backboard_width_ft / 2
    lines = [
        np.array([[half_l, lane], [ft_x, lane], [ft_x, -lane], [half_l, -lane]]),
        _arc(ft_x, 0.0, c.ft_circle_r_ft, 0, 2 * math.pi, 64),
        np.vstack([[half_l, corner], arc, [half_l, -corner]]),
        np.array([[bb_x, bb_w], [bb_x, -bb_w]]),
        _arc(rx, 0.0, c.rim_r_ft, 0, 2 * math.pi, 24),
    ]
    if c.restricted_r_ft:
        rr = c.restricted_r_ft
        lines.append(np.c_[rx - rr * np.cos(np.linspace(math.pi / 2, -math.pi / 2, 32)),
                           rr * np.sin(np.linspace(math.pi / 2, -math.pi / 2, 32))])  # fmt: skip
    return lines


def court_lines(court: str = "nba", half: bool = False) -> list[np.ndarray]:
    """Polylines (N, 2) in court feet for drawing. ``half`` draws the main half (x >= 0) only."""
    c = spec(court)
    half_l, half_w, cr = c.length_ft / 2, c.width_ft / 2, c.center_circle_r_ft
    if half:
        outline = np.array(
            [[0, half_w], [half_l, half_w], [half_l, -half_w], [0, -half_w], [0, half_w]], float
        )
        return [outline, _arc(0.0, 0.0, cr, -math.pi / 2, math.pi / 2, 32), *_half_lines(c)]
    outline = np.array(
        [[-half_l, half_w], [half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], float
    )
    main = _half_lines(c)
    return [
        outline,
        np.array([[0, half_w], [0, -half_w]], float),
        _arc(0.0, 0.0, cr, 0, 2 * math.pi, 64),
        *main,
        *[-pts for pts in main],
    ]


def fold_to_half(points_ft: np.ndarray) -> np.ndarray:
    """Rotate points on the other half (x < 0) onto the main half, as shot charts do."""
    p = np.asarray(points_ft, float).reshape(-1, 2).copy()
    other = p[:, 0] < 0
    p[other] *= -1
    return p


# ---------------------------------------------------------------- calibration


def _apply(h: np.ndarray, pts: Any) -> np.ndarray:
    p = np.asarray(pts, float)
    single = p.ndim == 1
    p = p.reshape(-1, 2)
    hom = np.c_[p, np.ones(len(p))] @ h.T
    out = hom[:, :2] / hom[:, 2:3]
    return out[0] if single else out


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT homography src -> dst (least squares over all points)."""

    def norm(p: np.ndarray) -> np.ndarray:
        mean, scale = p.mean(0), math.sqrt(2) / max(np.linalg.norm(p - p.mean(0), axis=1).mean(), 1e-12)
        return np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])

    ts, td = norm(src), norm(dst)
    s, d = _apply(ts, src), _apply(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


def _fit_court_to_image(ft: np.ndarray, img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    try:
        import cv2
    except ImportError:
        return _dlt(ft, img), np.ones(len(ft), bool)
    method = cv2.RANSAC if len(ft) > 4 else 0
    h, mask = cv2.findHomography(ft, img, method, RANSAC_PX)
    if h is None:
        return None, np.zeros(len(ft), bool)
    return h, (mask.ravel().astype(bool) if mask is not None else np.ones(len(ft), bool))


@dataclass
class Calibration:
    """Floor homography from image pixels (proxy) to court feet."""

    court: str
    H: np.ndarray  # 3x3, image px -> court ft
    image_points: list[tuple[float, float]]
    landmark_names: list[str]
    inliers: list[bool]
    reprojection_error_px: float
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_points(
        cls, image_points: list[tuple[float, float]], landmark_names: list[str], court: str = "nba"
    ) -> Calibration:
        if len(image_points) != len(landmark_names):
            raise ValueError("each clicked point needs exactly one landmark name")
        if len(image_points) < 4:
            raise ValueError("click at least 4 court landmarks (5+ lets us check the fit)")
        known = landmarks(court)
        if unknown := [n for n in landmark_names if n not in known]:
            raise ValueError(f"unknown landmark(s) {unknown}")
        if len(set(landmark_names)) != len(landmark_names):
            raise ValueError("each landmark can only be used once")
        if elevated := [n for n in landmark_names if n in ELEVATED]:
            raise ValueError(
                f"{elevated} is 10 ft above the floor; pick floor landmarks such as lane corners"
            )
        img = np.asarray(image_points, float).reshape(-1, 2)
        ft = np.asarray([known[n] for n in landmark_names], float)
        centered = ft - ft.mean(0)
        if np.linalg.svd(centered, compute_uv=False)[-1] < 1.0:
            raise ValueError("those landmarks lie on one line; add points off that line")
        h_ci, mask = _fit_court_to_image(ft, img)
        if h_ci is None or not np.all(np.isfinite(h_ci)) or abs(np.linalg.det(h_ci)) < 1e-12:
            raise ValueError("couldn't fit the court to those points; check the clicks")
        h = np.linalg.inv(h_ci)
        h /= h[2, 2]
        err = np.linalg.norm(_apply(h_ci, ft) - img, axis=1)
        rms = float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("inf")
        cal = cls(court, h, [tuple(map(float, p)) for p in img], list(landmark_names), mask.tolist(), rms)
        if not mask.all():
            bad = [n for n, ok in zip(landmark_names, mask, strict=True) if not ok]
            cal.warnings.append(f"ignored point(s) that didn't fit the others: {', '.join(bad)}")
        if len(img) == 4:
            cal.warnings.append("4 points always fit exactly; add a 5th to check accuracy")
        if cal.mirrored:
            cal.warnings.append("left and right look swapped (names are from half court facing the basket)")
        if rms > MAX_REPROJ_PX:
            cal.warnings.append(f"clicks disagree by {rms:.1f} px on average; re-check them")
        return cal

    def to_court(self, points_px: Any) -> np.ndarray:
        """Image pixels (N, 2) or (2,) -> court feet."""
        return _apply(self.H, points_px)

    def to_image(self, points_ft: Any) -> np.ndarray:
        """Court feet (N, 2) or (2,) -> image pixels."""
        return _apply(np.linalg.inv(self.H), points_ft)

    @property
    def mirrored(self) -> bool:
        """True when the fit flips the floor: a camera above the court never sees it mirrored.

        With image y pointing down and court y pointing left, a correct fit has det(Jacobian) < 0.
        """
        p = np.mean(self.image_points, axis=0)
        e = 0.5
        jx = (self.to_court(p + [e, 0]) - self.to_court(p - [e, 0])) / (2 * e)
        jy = (self.to_court(p + [0, e]) - self.to_court(p - [0, e])) / (2 * e)
        return float(jx[0] * jy[1] - jx[1] * jy[0]) > 0

    @property
    def is_valid(self) -> bool:
        w = np.c_[np.asarray(self.image_points), np.ones(len(self.image_points))] @ self.H[2]
        return bool(
            np.all(np.isfinite(self.H))
            and sum(self.inliers) >= 4
            and self.reprojection_error_px <= MAX_REPROJ_PX
            and (np.all(w > 0) or np.all(w < 0))  # every point on the same side of the horizon
            and not self.mirrored
        )

    def image_polygon(self, half: bool = False, margin_ft: float = 2.0) -> np.ndarray:
        """Court outline (plus a margin) in image pixels, e.g. for a supervision PolygonZone court mask."""
        c = spec(self.court)
        hl, hw = c.length_ft / 2 + margin_ft, c.width_ft / 2 + margin_ft
        x0 = -margin_ft if half else -hl
        return self.to_image(np.array([[x0, hw], [hl, hw], [hl, -hw], [x0, -hw]], float))

    def to_dict(self) -> dict[str, Any]:
        return {
            "court": self.court,
            "H": self.H.tolist(),
            "image_points": [list(p) for p in self.image_points],
            "landmark_names": list(self.landmark_names),
            "inliers": list(self.inliers),
            "reprojection_error_px": self.reprojection_error_px,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        return cls(
            court=d["court"],
            H=np.asarray(d["H"], float),
            image_points=[(float(x), float(y)) for x, y in d["image_points"]],
            landmark_names=list(d["landmark_names"]),
            inliers=[bool(b) for b in d["inliers"]],
            reprojection_error_px=float(d["reprojection_error_px"]),
            warnings=list(d.get("warnings", [])),
        )
