"""Fakes shared by System Two tests (a VisionLLM that records requests, and a synthetic AnalysisRun)."""

from __future__ import annotations

from hoops.reason.base import AnalyzeRequest, AnalyzeResult, Capabilities, Usage
from hoops.schemas import (
    AnalysisRun,
    ClipMeta,
    Event,
    EventType,
    NormalizationReport,
    Provenance,
    ShotInfo,
    StatLine,
    TeamRef,
    TeamSummary,
    VideoAsset,
)


class FakeVLM:
    key = "fake"

    def __init__(self, parsed=None, text: str = "", native_video: bool = False, max_images: int = 20):
        self.caps = Capabilities(
            native_video=native_video, max_images=max_images, max_px=2000, json_schema="strict"
        )
        self.parsed = parsed
        self.text = text
        self.requests: list[AnalyzeRequest] = []

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        self.requests.append(req)
        return AnalyzeResult(
            text=self.text,
            parsed=self.parsed,
            usage=Usage(100, 20),
            cost_usd=0.01,
            latency_ms=5.0,
            model_id="fake-1",
            provider="fake",
        )

    def list_models(self) -> list[str]:
        return ["fake-1"]


def _asset(path: str, duration: float) -> VideoAsset:
    return VideoAsset(
        path=path,
        width=640,
        height=360,
        fps="30",
        fps_float=30.0,
        n_frames=int(duration * 30),
        duration_s=duration,
    )


def _event(eid: str, etype: EventType, t: float, team: str, *, shot: ShotInfo | None = None, **kw) -> Event:
    return Event(
        event_id=eid,
        type=etype,
        t_start_s=t - 1.0,
        t_end_s=t + 0.5,
        t_key_s=t,
        team_id=team,
        shot=shot,
        confidence=kw.pop("confidence", 0.9),
        confidence_kind="heuristic",
        provenance=Provenance(source="rule", producer="test", stage="shots"),
        **kw,
    )


def make_run(duration: float = 60.0) -> AnalysisRun:
    clip = ClipMeta(
        clip_id="clip1",
        source_size_bytes=1,
        master=_asset("clips/clip1/master.mp4", duration),
        proxy=_asset("clips/clip1/proxy_720p.mp4", duration),
        normalization=NormalizationReport(summary="ok"),
    )
    events = [
        _event("e1", EventType.made_shot, 12.4, "A", shot=ShotInfo(points=2, result="made", kind="jumper"), status="confirmed"),
        _event("e2", EventType.missed_shot, 30.0, "B", shot=ShotInfo(points=3, result="missed", kind="jumper"), status="confirmed"),
        _event("e3", EventType.rebound, 31.0, "A", status="confirmed"),
        _event("e4", EventType.made_shot, 40.0, "B", shot=ShotInfo(points=2, result="made"), status="rejected"),
        _event("e5", EventType.three_pointer, 50.0, "A", shot=ShotInfo(points=3, result="made")),
    ]  # fmt: skip
    return AnalysisRun(
        run_id="run1",
        clip=clip,
        teams=[TeamRef(team_id="A", name="White"), TeamRef(team_id="B", name="Blue")],
        events=events,
        team_summaries=[
            TeamSummary(
                team=TeamRef(team_id="A", name="White"),
                stats=StatLine(pts=5, fga=2, fgm=2, tpa=1, tpm=1, reb=1),
            ),
            TeamSummary(team=TeamRef(team_id="B", name="Blue"), stats=StatLine(pts=0, fga=1, fgm=0, tpa=1)),
        ],
    )
