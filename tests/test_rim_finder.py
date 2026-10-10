from __future__ import annotations

import pandas as pd
import pytest

from hoops.config import ROOT
from hoops.perception.rim_finder import RimSuggestion, pick_active_rim, rim_from_hoop


def test_rim_from_hoop_trims_the_net():
    assert rim_from_hoop((100, 50, 200, 140)) == (100, 50, 200, 95)  # tall box -> top half (45 px)
    assert rim_from_hoop((100, 50, 200, 80)) == (100, 50, 200, 80)  # already rim-shaped


def _balls(points):
    return pd.DataFrame(
        [{"cls": "ball", "x1": x - 5, "y1": y - 5, "x2": x + 5, "y2": y + 5} for x, y in points]
    )


def test_pick_active_rim_follows_the_ball():
    near = RimSuggestion(box=(140, 10, 260, 50), score=0.6, hoop_box=(140, 10, 260, 110))
    played = RimSuggestion(box=(760, 250, 820, 275), score=0.4, hoop_box=(760, 250, 820, 300))
    arc = [(700 + i * 5, 400 - i * 25) for i in range(10)]  # ball rising toward the far-right hoop
    assert pick_active_rim([near, played], _balls(arc)) is played
    assert pick_active_rim([near, played], pd.DataFrame(columns=["cls", "x1", "y1", "x2", "y2"])) is near
    assert pick_active_rim([], _balls(arc)) is None


@pytest.mark.slow
def test_suggest_rims_on_sample_poster():
    from hoops.frames import frames_at
    from hoops.perception.rim_finder import suggest_rims

    clip = ROOT / "data" / "samples" / "plos_shot_04.ogv"
    if not clip.exists():
        pytest.skip("run scripts/fetch_samples.py")
    ((_, rgb),) = frames_at(clip, [0.3])
    sugg = suggest_rims(rgb)
    assert sugg and sugg[0].score > 0.2


def test_same_hoop_merges_panned_boxes():
    from hoops.perception.rim_finder import same_hoop

    assert same_hoop((779, 237, 830, 261), (765, 255, 819, 278))  # one hoop, camera moved
    assert not same_hoop((335, 62, 450, 104), (1088, 172, 1155, 205))  # two different hoops
