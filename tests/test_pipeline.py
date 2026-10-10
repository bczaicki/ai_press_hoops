"""Pipeline orchestration with a scripted detector and a fake VLM: no weights, no network, no cost."""

from __future__ import annotations

import pytest

from hoops import config, registry
from hoops.config import ProviderProfile
from hoops.pipeline import RunRequest, VLMChoice, estimate, run_analysis, stages_for
from hoops.reason.base import AnalyzeResult, Capabilities, Usage
from hoops.schemas import ClipMeta, ClipSummary, KeyMoment, NormalizationReport, VideoAsset, WindowVerdict
from hoops.store import Store
from tests.helpers import make_video, requires_ffmpeg

RIM = (600.0, 200.0, 660.0, 222.0)


def _ball_at(t: float) -> tuple[float, float] | None:
    """A jump shot: rises from (300, 500) to an apex above the rim, drops through it at t~1.6 s."""
    if not 0.4 <= t <= 2.2:
        return None
    u = (t - 0.4) / 1.2  # 0 at release, 1 at the rim
    x = 300 + (630 - 300) * u
    y = 500 - 700 * u + 410 * u * u  # apex above the rim, then down through y~211 at u=1
    return x, y


def _script(frame: int, t: float):
    from hoops.perception.detectors import dets_from_rows

    rows = [("person", 0.9, 250.0, 380.0, 330.0, 620.0)]
    pos = _ball_at(t)
    if pos:
        x, y = pos
        rows.append(("ball", 0.8, x - 9, y - 9, x + 9, y + 9))
    return dets_from_rows(rows)


class FakeVLM:
    key = "fake"
    caps = Capabilities(native_video=False, max_images=20, max_px=1280, json_schema="strict")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, req):
        self.calls.append(req.purpose)
        parsed = None
        if req.output_model is WindowVerdict:
            parsed = WindowVerdict(verdict="made", shot_kind="jumper", confidence=0.9, evidence_tiles=["W1#3"],
                                   shooter_description="white jersey", reason="ball drops through the net")
        elif req.output_model is ClipSummary:
            parsed = ClipSummary(headline="One jumper, one make", narrative="A clean jump shot goes in.",
                                 key_moments=[KeyMoment(ref="0:01.6", description="make")], caveats=[])
        return AnalyzeResult(text=parsed.model_dump_json() if parsed else "ok", parsed=parsed, usage=Usage(10, 5),
                             cost_usd=0.001, latency_ms=5.0, model_id="fake-1", provider="fake", raw={"ok": True})

    def list_models(self):
        return ["fake-1"]


@pytest.fixture
def fake_world(data_dir, monkeypatch, tmp_path):
    pytest.importorskip("hoops.perception.detectors")
    pytest.importorskip("hoops.events.escalate")
    pytest.importorskip("hoops.reason.summarize")
    from hoops.perception.detectors import ScriptedDetector

    registry.load_plugins()
    registry.detectors.register("fake:shot", label="scripted shot")(lambda conf=0.35: ScriptedDetector(_script))
    vlm = FakeVLM()
    registry.vlms.register("fake", label="fake")(lambda: vlm)
    monkeypatch.setitem(config.PROFILES, "fake", ProviderProfile(
        key="fake", label="Fake", kind="openai_compat", default_model="fake-1", key_field=None,
        sends_data_to="nobody (test)", local=True))
    store = Store(data_dir)
    clip_id = "feedfacecafebeef"
    d = store.clip_dir(clip_id)
    make_video(d / "master.mp4", duration=3.0, size="1280x720", fps="30")
    make_video(d / "proxy_720p.mp4", duration=3.0, size="1280x720", fps="30", audio=False)
    make_video(tmp_path / "unused.mp4", duration=0.2)

    def asset(name: str) -> VideoAsset:
        return VideoAsset(path=f"clips/{clip_id}/{name}", width=1280, height=720, fps="30", fps_float=30.0,
                          n_frames=90, duration_s=3.0, has_audio=name == "master.mp4")

    clip = ClipMeta(clip_id=clip_id, source_size_bytes=1, master=asset("master.mp4"),
                    proxy=asset("proxy_720p.mp4"), normalization=NormalizationReport(summary="test"),
                    footage_type="fixed_camera")
    store.save_clip(clip)
    return store, clip, vlm


@requires_ffmpeg
def test_full_run_with_fakes(fake_world):
    store, clip, vlm = fake_world
    req = RunRequest(clip_id=clip.clip_id, outputs=["annotated_video", "events", "summary", "court_map"],
                     detector="fake:shot", tracker="bytetrack", decider="rules", rim_box=list(RIM),
                     vlm=VLMChoice(provider="fake"), egress_ok=["fake"])
    assert stages_for(req) == ["detect", "track", "shots", "decide", "adjudicate", "summarize", "court", "render"]
    run = run_analysis(req, None, store)
    assert run.status == "done"
    shots = [e for e in run.events if e.shot is not None]
    assert shots, f"expected a shot event; warnings={run.warnings}"
    assert run.annotated_video_path and store.abs(run.annotated_video_path).exists()
    assert run.captions_path and run.court_map_paths
    assert run.narratives and "summary" in vlm.calls
    assert run.decisions, "System One should have logged decisions"
    assert store.load_run(run.run_id).events == run.events


def test_estimate_and_stages_without_vlm(data_dir):
    store = Store(data_dir)
    d = store.clip_dir("abcdefabcdefabcd")
    asset = VideoAsset(path=f"clips/abcdefabcdefabcd/master.mp4", width=1280, height=720, fps="30", fps_float=30.0,
                       n_frames=900, duration_s=30.0)
    clip = ClipMeta(clip_id="abcdefabcdefabcd", source_size_bytes=1, master=asset, proxy=asset,
                    normalization=NormalizationReport())
    del d
    req = RunRequest(clip_id=clip.clip_id, outputs=["summary"])
    assert stages_for(req) == []
    est = estimate(req, clip)
    assert est.estimate_usd == 0.0
