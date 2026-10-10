"""Pydantic schemas: the single source of truth for runs, events, and VLM I/O.

Two families:
- Canonical models (``M``): what we store in ``data/runs/<id>/run.json``. May use defaults and dicts.
- VLM-facing models (``VM``): what we ask providers to emit. Flat and strict-schema friendly for both
  Anthropic and OpenAI: every field required (nullable instead of optional), extra forbidden, no numeric
  bounds / length limits / dicts / recursion. Validate ranges in code after parsing.

Canonical time is seconds on the master timeline. VLMs cite tile ids burned into contact sheets
(e.g. ``S2#4``); code maps them to seconds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


class M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventType(str, Enum):
    shot_attempt = "shot_attempt"
    made_shot = "made_shot"
    missed_shot = "missed_shot"
    three_pointer = "three_pointer"
    free_throw = "free_throw"
    rebound = "rebound"
    assist = "assist"
    turnover = "turnover"
    steal = "steal"
    block = "block"
    foul = "foul"
    possession_change = "possession_change"
    timeout = "timeout"
    other = "other"


FootageType = Literal[
    "broadcast", "fixed_camera", "handheld", "highlights", "practice", "not_basketball", "unknown"
]
EventStatus = Literal["provisional", "confirmed", "rejected", "needs_review"]
Source = Literal["cv", "rule", "system_one", "vlm", "human", "import"]
ShotKind = Literal["jumper", "layup", "dunk", "hook", "tip", "free_throw", "unknown"]
OutputKind = Literal["annotated_video", "events", "summary", "court_map"]


# ---------------------------------------------------------------- canonical: provenance & clip


class Provenance(M):
    source: Source
    producer: str  # e.g. "rfdetr==1.11.2:coco-small", "anthropic:claude-opus-5-5", "typesafe:jev-latest"
    stage: str  # ingest | detect | track | shots | decide | adjudicate | summarize | qa | review
    prompt_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Evidence(M):
    t_s: float
    frame: int | None = None
    image_path: str | None = None
    tile_id: str | None = None
    note: str | None = None


class VideoAsset(M):
    path: str  # relative to settings.data_dir
    width: int
    height: int
    fps: str  # rational, e.g. "30000/1001"
    fps_float: float
    n_frames: int
    duration_s: float
    codec: str = "h264"
    pix_fmt: str = "yuv420p"
    has_audio: bool = False


class NormalizationReport(M):
    rotation_deg: int = 0
    hdr: Literal["none", "hlg", "pq", "dolby_vision"] = "none"
    tonemapped: bool = False
    deinterlaced: bool = False
    vfr_source: bool = False
    sar_fixed: bool = False
    fps_snapped_from: str | None = None
    source_resolution: tuple[int, int] | None = None
    ffmpeg_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""  # plain English, e.g. "HDR->SDR, rotated 90 deg, 29.97 fps CFR"


class ClipMeta(M):
    clip_id: str  # sha256(source bytes)[:16]
    original_filename: str | None = None
    source_size_bytes: int
    master: VideoAsset
    proxy: VideoAsset
    normalization: NormalizationReport
    footage_type: FootageType = "unknown"
    footage_type_source: Source | None = None
    scene_cuts_s: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------- canonical: teams, tracks, events


class TeamRef(M):
    team_id: str  # "A"/"B" until the user names them
    name: str | None = None
    color_hex: str | None = None


class PlayerRef(M):
    player_id: str  # local: "A#23" or "trk_17"
    team_id: str | None = None
    jersey_number: str | None = None
    name: str | None = None  # only from a user-supplied roster (never face recognition)
    track_ids: list[int] = Field(default_factory=list)


class Track(M):
    track_id: int
    cls: str
    first_t_s: float
    last_t_s: float
    n_frames: int
    mean_score: float
    team_id: str | None = None
    player_id: str | None = None


class ShotInfo(M):
    points: Literal[1, 2, 3] | None = None
    result: Literal["made", "missed", "unknown"] = "unknown"
    kind: ShotKind = "unknown"
    court_x_ft: float | None = None  # after homography; origin = center court
    court_y_ft: float | None = None
    image_x: float | None = None  # shooter location in proxy pixels (fallback when no court map)
    image_y: float | None = None


class Event(M):
    event_id: str
    type: EventType
    t_start_s: float
    t_end_s: float | None = None
    t_key_s: float | None = None  # release, rim contact, whistle
    team_id: str | None = None
    player_ids: list[str] = Field(default_factory=list)
    related_event_ids: list[str] = Field(default_factory=list)
    shot: ShotInfo | None = None
    is_replay: bool = False
    confidence: float = Field(ge=0, le=1)
    confidence_kind: Literal["model_score", "calibrated", "self_reported", "heuristic", "consensus", "human"]
    status: EventStatus = "provisional"
    review: Literal["unreviewed", "accepted", "rejected", "edited"] = "unreviewed"
    description: str | None = None
    # Raw numbers + qualitative descriptors + per-layer scores, kept so thresholds can be re-applied
    # (Strict <-> Eager) without re-running inference.
    features: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    provenance: Provenance


class StatLine(M):
    pts: int = 0
    fga: int = 0
    fgm: int = 0
    tpa: int = 0
    tpm: int = 0
    fta: int = 0
    ftm: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0


class TeamSummary(M):
    team: TeamRef
    stats: StatLine
    derived_from_event_ids: list[str] = Field(default_factory=list)


class PlayerSummary(M):
    player: PlayerRef
    stats: StatLine
    on_screen_s: float | None = None
    derived_from_event_ids: list[str] = Field(default_factory=list)


class Narrative(M):
    kind: Literal["recap", "play_by_play", "coaching", "qa", "custom"]
    text: str
    question: str | None = None
    cited_event_ids: list[str] = Field(default_factory=list)
    cited_times_s: list[float] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)  # flagged by the System One claim check
    provenance: Provenance


# ---------------------------------------------------------------- canonical: audit records


class DecisionRecord(M):
    """One System One call (Jev / OpenAI Decisions / adapter / rules)."""

    decision_id: str
    decider: str  # "jev", "openai-decisions", "adapter:anthropic:claude-opus-5-5", "rules"
    purpose: Literal["triage", "event_typing", "escalation", "qa_route", "claim_check"]
    subject_id: str | None = None  # event / window id
    state: dict[str, Any] | str
    questions: dict[str, Any]  # serialized question specs
    answers: dict[str, Any]  # name -> {kind, value, probabilities, confidence}
    routed: Literal["accept", "reject", "escalate"] | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class VLMCallRecord(M):
    call_id: str
    provider: str  # profile key
    model: str
    purpose: Literal["summary", "adjudicate", "qa", "caption", "triage", "claim_check", "custom"]
    n_images: int = 0
    video_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    stop_reason: str | None = None
    raw_path: str | None = None  # data/runs/<id>/raw_vlm/<call_id>.json
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class StageConfig(M):
    stage: str
    impl: str  # registry key, e.g. "rfdetr:coco"
    version: str | None = None
    license: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(M):
    wall_s: float = 0.0
    detect_fps: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_estimate_usd: float | None = None


class AnalysisRun(M):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)
    clip: ClipMeta
    outputs_requested: list[OutputKind] = Field(default_factory=list)
    pipeline: list[StageConfig] = Field(default_factory=list)
    data_egress: list[str] = Field(default_factory=list)  # providers that received frames/video this run
    teams: list[TeamRef] = Field(default_factory=list)
    players: list[PlayerRef] = Field(default_factory=list)
    detections_path: str | None = None  # detections.parquet, relative to data_dir
    annotated_video_path: str | None = None
    captions_path: str | None = None  # WebVTT
    court_map_paths: list[str] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    team_summaries: list[TeamSummary] = Field(default_factory=list)
    player_summaries: list[PlayerSummary] = Field(default_factory=list)
    narratives: list[Narrative] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    vlm_calls: list[VLMCallRecord] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["running", "done", "failed", "cancelled"] = "running"
    error: str | None = None


# Columns of data/runs/<id>/detections.parquet (one row per object per analyzed frame, proxy pixels).
DETECTION_COLUMNS = (
    "frame",  # proxy frame index
    "t_s",  # seconds on the master timeline
    "cls",  # normalized class name (hoops.perception.base.CLASS_VOCAB)
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",  # -1 when untracked
    "team_id",  # "" when unknown
    "interpolated",  # bool: ball positions filled across short gaps
)


# ---------------------------------------------------------------- VLM-facing (strict, flat)


class VM(BaseModel):
    """Base for provider-emitted models. No defaults: every field is required (nullable when optional)."""

    model_config = ConfigDict(extra="forbid")


class VLMEvent(VM):
    type: EventType
    start_ref: str  # tile id "S2#4" or "MM:SS.s"; resolved to seconds in code
    end_ref: str | None
    team_label: str | None  # as seen, e.g. "white jerseys"
    player_label: str | None  # e.g. "#23 white"
    shot_points: int | None  # 1/2/3, checked in code
    shot_result: Literal["made", "missed", "unknown"] | None
    is_replay: bool
    confidence: float  # 0-1, checked in code
    evidence: str  # one sentence citing tiles


class VLMResult(VM):
    events: list[VLMEvent]
    narrative: str
    unsure_about: list[str]


class KeyMoment(VM):
    ref: str  # tile id or MM:SS
    description: str


class ClipSummary(VM):
    headline: str
    narrative: str
    key_moments: list[KeyMoment]
    caveats: list[str]


class WindowVerdict(VM):
    """System Two adjudication of one escalated candidate window."""

    verdict: Literal["made", "missed", "no_shot", "unclear"]
    shot_kind: ShotKind
    confidence: float  # 0-1, checked in code
    evidence_tiles: list[str]
    shooter_description: str | None  # jersey color/number as seen; never a name
    reason: str
