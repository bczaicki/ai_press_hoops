from __future__ import annotations

from hoops.reason.summarize import STRUCTURED_FORMAT, adjudicate, summarize
from hoops.schemas import ClipSummary, KeyMoment, WindowVerdict
from tests.helpers import requires_ffmpeg
from tests.reason_fakes import FakeVLM

SUMMARY = ClipSummary(
    headline="White closes strong",
    narrative="White hit two late shots.",
    key_moments=[
        KeyMoment(ref="S1#2", description="Pull-up jumper"),
        KeyMoment(ref="01:05", description="Stop"),
    ],
    caveats=["Ball hidden in S1#3"],
)


@requires_ffmpeg
def test_summarize_with_sheets(testsrc_video):
    vlm = FakeVLM(parsed=SUMMARY)
    summary, text, res = summarize(
        vlm, video=testsrc_video, duration_s=2.0, fps=2.0, cv_context='{"shots": 2}'
    )
    req = vlm.requests[0]
    assert req.output_model is ClipSummary and req.purpose == "summary" and req.effort == "medium"
    assert req.video is None and len(req.images) == 1 and req.images[0].label.startswith("Sheet S1")
    assert STRUCTURED_FORMAT in req.prompt and "recap" in req.prompt.lower() and "tile id" in req.system
    assert summary.key_moments[0].ref.startswith("S1#2 @ 00:0")  # tile resolved to a clip time
    assert summary.key_moments[1].ref == "01:05"
    assert text.startswith("**White closes strong**") and "Caveats:" in text
    assert res.parsed is summary


def test_summarize_native_video_and_free_text(tmp_path):
    vlm = FakeVLM(text="Free-form recap.", native_video=True)
    video = tmp_path / "clip.mp4"
    summary, text, _ = summarize(
        vlm, video=video, duration_s=30.0, native_video=True, fps=2.0, structured=False, preset="coach"
    )
    req = vlm.requests[0]
    assert summary is None and text == "Free-form recap."
    assert req.images == [] and req.video.path == video and req.video.fps == 2.0
    assert req.output_model is None and "MM:SS" in req.prompt and "coach" in req.prompt.lower()


def test_custom_prompt():
    vlm = FakeVLM(text="ok", native_video=True)
    summarize(
        vlm,
        video="x.mp4",
        duration_s=10,
        native_video=True,
        preset="custom",
        custom_prompt="Count dunks only.",
    )
    assert vlm.requests[0].prompt.startswith("Count dunks only.")


@requires_ffmpeg
def test_adjudicate_frames_with_rim_crops(testsrc_video):
    verdict = WindowVerdict(
        verdict="made",
        shot_kind="jumper",
        confidence=1.4,
        evidence_tiles=["R1#5"],
        shooter_description=None,
        reason="Through the net.",
    )
    vlm = FakeVLM(parsed=verdict)
    out, _res = adjudicate(
        vlm, video=testsrc_video, t0=0.2, t1=1.8, cv_context="{}", rim_box=(300, 150, 340, 180)
    )
    req = vlm.requests[0]
    assert req.effort == "low" and req.output_model is WindowVerdict and req.purpose == "adjudicate"
    labels = [i.label for i in req.images]
    assert labels[0].startswith("Sheet R1") and labels[-1].startswith("Sheet W1")
    assert "00:00.20" in req.prompt and out.confidence == 1.0


def test_adjudicate_native_video_window(tmp_path):
    vlm = FakeVLM(parsed=None, text="", native_video=True)
    out, _ = adjudicate(vlm, video=tmp_path / "m.mp4", t0=10.0, t1=13.5, cv_context="{}")
    v = vlm.requests[0].video
    assert out is None and (v.t0_s, v.t1_s, v.fps) == (10.0, 13.5, 8.0) and vlm.requests[0].images == []
