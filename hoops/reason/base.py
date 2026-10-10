"""System Two contracts: VLM providers behind one interface.

Every adapter takes an ``AnalyzeRequest`` and returns an ``AnalyzeResult``. Adapters own provider
specifics (SDK calls, structured-output mechanics, image encoding, retries, refusal handling, cost
from usage). The pipeline/UI only see these types plus the profile's capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Purpose = Literal["summary", "adjudicate", "qa", "caption", "triage", "claim_check", "custom"]


@dataclass(frozen=True)
class Capabilities:
    native_video: bool
    max_images: int
    max_px: int
    json_schema: Literal["strict", "loose", "none"]
    audio: bool = False
    local: bool = False


@dataclass
class ImagePart:
    jpeg: bytes
    label: str  # sent as a text block right before the image, e.g. "S2#4 t=12.40s"
    t_s: float | None = None
    tile_id: str | None = None


@dataclass
class VideoPart:
    path: Path  # a normalized H.264 mp4 (master, proxy, or a trimmed window)
    t0_s: float | None = None
    t1_s: float | None = None
    fps: float | None = None  # requested sampling fps (native-video providers only)


@dataclass
class AnalyzeRequest:
    prompt: str
    system: str | None = None
    images: list[ImagePart] = field(default_factory=list)
    video: VideoPart | None = None
    cv_context: str | None = None  # compact JSON/text from CV (tracks, candidate events)
    output_model: type[BaseModel] | None = None  # a hoops.schemas.VM subclass for structured output
    history: list[tuple[str, str]] = field(default_factory=list)  # prior (user, assistant) Q&A turns
    effort: Effort = "medium"
    max_tokens: int = 16000
    purpose: Purpose = "custom"
    model: str | None = None  # override the profile default
    cache_key: str | None = None  # stable id (e.g. run_id) so adapters can reuse uploads / prompt cache


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class AnalyzeResult:
    text: str
    parsed: BaseModel | None
    usage: Usage
    cost_usd: float | None
    latency_ms: float
    model_id: str
    provider: str  # profile key
    stop_reason: str | None = None
    refused: bool = False
    error: str | None = None
    raw: Any = None  # JSON-serializable provider response (saved under raw_vlm/)


@runtime_checkable
class VisionLLM(Protocol):
    key: str  # profile key, e.g. "claude"
    caps: Capabilities

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult: ...

    def list_models(self) -> list[str]:
        """Model ids available to the configured key (for the UI dropdown)."""
        ...
