from __future__ import annotations

import numpy as np

from hoops.frames import frames_at
from hoops.perception.detectors import ScriptedDetector
from hoops.perception.run import run_perception
from hoops.render import flash_label, render_annotated
from hoops.schemas import Event, EventType, Provenance, ShotInfo
from hoops.video_io import ffprobe_json, is_browser_playable
from tests.helpers import make_video, requires_ffmpeg

pytestmark = requires_ffmpeg


def script(frame: int, t_s: float) -> list[tuple]:
    x = 60 + 3 * frame
    rows = [
        ("player", 0.9, x, 180, x + 40, 300),
        ("player", 0.9, 400, 170, 440, 290),
        ("referee", 0.9, 520, 170, 560, 290),
        ("rim", 0.9, 300, 80, 330, 88),
    ]
    if frame % 7:
        rows.append(("ball", 0.8, 200 + 3 * frame, 120, 212 + 3 * frame, 132))
    return rows


def event(kind: EventType, t: float, result: str = "unknown", status: str = "provisional") -> Event:
    return Event(
        event_id=f"e{t}",
        type=kind,
        t_start_s=t - 0.5,
        t_key_s=t,
        shot=ShotInfo(result=result),  # type: ignore[arg-type]
        confidence=0.9,
        confidence_kind="heuristic",
        status=status,  # type: ignore[arg-type]
        provenance=Provenance(source="cv", producer="test", stage="shots"),
    )


def test_render_annotated_is_browser_playable_with_audio(tmp_path):
    clip = make_video(tmp_path / "proxy.mp4", duration=2.0, size="640x360", fps="30", audio=True)
    res = run_perception(clip, detector=ScriptedDetector(script))
    df = res.detections.copy()
    df.loc[(df["cls"] == "player") & (df["track_id"] == df["track_id"].max()), "team_id"] = "A"
    seen: list[float] = []
    out = render_annotated(
        clip,
        df,
        tmp_path / "annotated.mp4",
        events=[event(EventType.made_shot, 1.0, "made")],
        team_colors={"A": "#f5f5f5", "B": "#1e3a8a"},
        master_audio=clip,
        progress=lambda p, m: seen.append(p),
    )
    assert out.exists() and is_browser_playable(out)
    streams = ffprobe_json(out)["streams"]
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    assert not (tmp_path / "annotated.video.mp4").exists()
    assert seen[-1] == 1.0

    before, after = frames_at(clip, [1.0])[0][1], frames_at(out, [1.0])[0][1]
    assert np.abs(before.astype(int) - after.astype(int)).mean() > 1  # overlays were drawn


def test_render_without_detections_or_audio(tmp_path):
    clip = make_video(tmp_path / "proxy.mp4", duration=1.0, size="320x180", fps="15", audio=False)
    res = run_perception(clip, detector=ScriptedDetector({}))
    out = render_annotated(clip, res.detections, tmp_path / "plain.mp4", rim_box=(10, 10, 40, 20))
    assert is_browser_playable(out)


def test_flash_labels():
    assert flash_label(event(EventType.made_shot, 1)) == "MADE"
    assert flash_label(event(EventType.shot_attempt, 1, "missed")) == "MISS"
    assert flash_label(event(EventType.three_pointer, 1)) == "SHOT"
    assert flash_label(event(EventType.rebound, 1)) is None
    assert flash_label(event(EventType.made_shot, 1, status="rejected")) is None
