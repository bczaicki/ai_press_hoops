"""Team split: torso-colour k-means (fast, no model), SigLIP ``TeamClassifier`` (roboflow/sports), or two
user-picked jersey colours. Referees are never assigned a team.

Team ids are "A"/"B" until the user names them. With k-means/SigLIP, "A" is the lighter-jersey team.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from hoops.frames import iter_frames
from hoops.schemas import TeamRef

log = logging.getLogger(__name__)

TEAMABLE = ("player", "person")
TEAM_IDS = ("A", "B")
MIN_BOX_H = 24  # px; smaller boxes give mostly floor colour


def torso(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Upper-middle of a person box (the jersey): 25-75% of the width, 15-50% of the height."""
    h_img, w_img = rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    left, right = int(max(0, x1 + 0.25 * w)), int(min(w_img, x2 - 0.25 * w))
    top, bottom = int(max(0, y1 + 0.15 * h)), int(min(h_img, y1 + 0.5 * h))
    return rgb[top:bottom, left:right]


def color_feature(rgb_crop: np.ndarray) -> np.ndarray:
    """[S*cos(H), S*sin(H), V] medians: hue separates red/blue, value separates white/black."""
    hsv = np.asarray(Image.fromarray(np.ascontiguousarray(rgb_crop)).convert("HSV"), dtype=float) / 255.0
    hue = hsv[..., 0] * 2 * np.pi
    sat, val = hsv[..., 1], hsv[..., 2]
    return np.array([np.median(sat * np.cos(hue)), np.median(sat * np.sin(hue)), np.median(val)])


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = (round(float(v)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def kmeans2(x: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-means: seeds are the point farthest from the mean and the point farthest from it."""
    c0 = x[np.argmax(((x - x.mean(0)) ** 2).sum(1))]
    c1 = x[np.argmax(((x - c0) ** 2).sum(1))]
    centers = np.stack([c0, c1]).astype(float)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.stack([x[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(2)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels, centers


def _sample(people: pd.DataFrame, per_track: int) -> pd.DataFrame:
    picks = []
    for _, g in people.groupby("track_id", sort=True):
        big = g[(g["y2"] - g["y1"]) >= MIN_BOX_H]
        g = (big if len(big) else g).sort_values("frame")
        idx = np.unique(np.linspace(0, len(g) - 1, num=min(per_track, len(g))).round().astype(int))
        picks.append(g.iloc[idx])
    return pd.concat(picks) if picks else people.iloc[:0]


def _crops(proxy_path: Path, rows: pd.DataFrame) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """(track ids, torso crops, full-body crops) for the sampled rows, in one decode pass."""
    wanted: dict[int, list[tuple]] = {}
    for r in rows.itertuples(index=False):
        wanted.setdefault(int(r.frame), []).append((int(r.track_id), (r.x1, r.y1, r.x2, r.y2)))
    last, t_end = max(wanted), float(rows["t_s"].max()) + 0.5
    tids, torsos, bodies = [], [], []
    for i, (_, rgb) in enumerate(iter_frames(proxy_path, t1=t_end)):
        for tid, box in wanted.get(i, ()):
            t = torso(rgb, box)
            if t.size == 0 or min(t.shape[:2]) < 2:
                continue
            x1, y1, x2, y2 = (round(v) for v in box)
            tids.append(tid)
            torsos.append(t)
            bodies.append(rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)])
        if i >= last:
            break
    return tids, torsos, bodies


def _siglip_labels(bodies: list[np.ndarray]) -> np.ndarray:
    from sports.common.team import TeamClassifier

    from hoops.perception.detectors import pick_device

    bgr = [np.ascontiguousarray(b[..., ::-1]) for b in bodies]  # TeamClassifier converts BGR -> RGB itself
    try:
        clf = TeamClassifier(device=pick_device())
        clf.fit(bgr)
        return np.asarray(clf.predict(bgr), dtype=int)
    except RuntimeError:  # an MPS op gap: retry on CPU
        clf = TeamClassifier(device="cpu")
        clf.fit(bgr)
        return np.asarray(clf.predict(bgr), dtype=int)


def assign_teams(
    proxy_path: Path,
    detections: pd.DataFrame,
    *,
    method: Literal["hsv", "siglip"] = "hsv",
    manual_colors: tuple[str, str] | None = None,
    max_crops_per_track: int = 8,
) -> tuple[pd.DataFrame, list[TeamRef]]:
    """Fill ``team_id`` ("A"/"B") for tracked players; returns (new detections, team refs)."""
    df = detections.copy()
    people = df[df["cls"].isin(TEAMABLE) & (df["track_id"] >= 0)]
    if people["track_id"].nunique() < (1 if manual_colors else 2):
        return df, []
    tids, torsos, bodies = _crops(Path(proxy_path), _sample(people, max_crops_per_track))
    if len(set(tids)) < (1 if manual_colors else 2):
        return df, []
    feats = np.stack([color_feature(t) for t in torsos])

    if manual_colors:
        refs = np.stack(
            [color_feature(np.full((4, 4, 3), hex_to_rgb(c), dtype=np.uint8)) for c in manual_colors]
        )
        labels = np.argmin(((feats[:, None, :] - refs[None]) ** 2).sum(-1), axis=1)
        order = [0, 1]
    else:
        if method == "siglip" and len(bodies) >= 16:
            labels = _siglip_labels(bodies)
        else:
            if method == "siglip":
                log.warning("too few crops (%d) for SigLIP team split; using colour k-means", len(bodies))
            labels, _ = kmeans2(feats)
        brightness = [feats[labels == k, 2].mean() if (labels == k).any() else -1.0 for k in range(2)]
        order = list(np.argsort(brightness)[::-1])  # lighter cluster -> "A"

    team_of_cluster = {int(k): TEAM_IDS[rank] for rank, k in enumerate(order)}
    votes: dict[int, Counter] = {}
    for tid, lab in zip(tids, labels, strict=True):
        votes.setdefault(tid, Counter())[team_of_cluster[int(lab)]] += 1
    track_team = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

    tracked_people = df["cls"].isin(TEAMABLE) & (df["track_id"] >= 0)
    df.loc[tracked_people, "team_id"] = df.loc[tracked_people, "track_id"].map(track_team).fillna("")

    teams = []
    for rank, team in enumerate(TEAM_IDS):
        members = [t for t, tid in zip(torsos, tids, strict=True) if track_team.get(tid) == team]
        if manual_colors:
            color = manual_colors[rank]
        elif members:
            pixels = np.concatenate([m.reshape(-1, 3) for m in members])
            color = rgb_to_hex(np.median(pixels, axis=0))
        else:
            continue
        teams.append(TeamRef(team_id=team, color_hex=color))
    return df, teams


def track_teams(detections: pd.DataFrame) -> dict[int, str]:
    """track_id -> team_id for tracks that have one (to fill ``Track.team_id``)."""
    rows = detections[(detections["track_id"] >= 0) & (detections["team_id"] != "")]
    return {
        int(k): str(v) for k, v in rows.groupby("track_id")["team_id"].agg(lambda s: s.mode().iloc[0]).items()
    }
