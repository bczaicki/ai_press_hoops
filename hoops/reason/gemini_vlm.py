"""Gemini via ``google-genai`` 2.29: the Interactions API (``client.interactions.create``), native video.

- Video parts: ``{"type": "video", "data"|"uri", "mime_type", "processing": {"type": "static", "fps": ...}}``.
  Windows are trimmed with ffmpeg into a small mp4 first, so we never rely on clip offsets (the docs disagree on
  their units; the 2.29 SDK docstring says "10.5s" strings, unverified live).
- Small clips go inline (base64); larger ones are uploaded with the Files API and polled until ACTIVE (files
  expire after 48 h, so uploads are cached for 47 h per ``cache_key``).
- Structured output: ``response_format={"type": "text", "mime_type": "application/json", "schema": ...}``.
- Paid tier only (the free tier may train on content) - that's a key/billing setting, not something code can check.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from hoops.config import PROFILES, get_settings
from hoops.cost import cost_from_usage
from hoops.reason.base import AnalyzeRequest, AnalyzeResult, Capabilities, Usage
from hoops.reason.common import (
    EFFORT_3,
    b64,
    elapsed_ms,
    failure,
    fit_jpeg,
    history_text,
    inline_refs,
    limit_images,
    now,
    parse_output,
    short_error,
    to_jsonable,
    trim_clip,
    user_text,
)
from hoops.registry import vlms

PROFILE = PROFILES["gemini"]
INLINE_MAX_BYTES = 20 * 1024 * 1024
FILE_TTL_S = 47 * 3600
_UPLOADS: dict[str, tuple[str, str, float]] = {}  # key -> (uri, mime_type, uploaded_at)
_UPLOAD_LOCK = threading.Lock()


def _state(f: Any) -> str:
    s = getattr(f, "state", None)
    return str(getattr(s, "name", s) or "").upper()


class GeminiVLM:
    key = "gemini"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any = None,
        poll_s: float = 2.0,
        upload_timeout_s: float = 300.0,
    ):
        self.model = model or PROFILE.default_model
        self.caps = Capabilities(
            native_video=True,
            max_images=PROFILE.max_images,
            max_px=PROFILE.max_px,
            json_schema="strict",
            audio=True,
        )
        if client is None:
            from google import genai

            client = genai.Client(api_key=get_settings().key("gemini_api_key"))
        self.client = client
        self.poll_s = poll_s
        self.upload_timeout_s = upload_timeout_s

    # ------------------------------------------------------------------ video
    def _upload(self, path: Path) -> tuple[str, str]:
        f = self.client.files.upload(file=str(path))
        deadline = time.monotonic() + self.upload_timeout_s
        while _state(f) == "PROCESSING":
            if time.monotonic() > deadline:
                raise TimeoutError("Gemini is still processing the uploaded video; try again shortly")
            time.sleep(self.poll_s)
            f = self.client.files.get(name=f.name)
        if _state(f) not in ("ACTIVE", ""):
            raise RuntimeError(f"Gemini could not process the video (state {_state(f)})")
        return f.uri, getattr(f, "mime_type", None) or "video/mp4"

    def _video_part(self, req: AnalyzeRequest) -> dict:
        v = req.video
        assert v is not None
        path, trimmed = Path(v.path), False
        if v.t0_s is not None or v.t1_s is not None:
            start = v.t0_s or 0.0
            path, trimmed = trim_clip(path, start, v.t1_s if v.t1_s is not None else start + 1e6), True
        processing = {"type": "static", "fps": float(v.fps or 1.0)}
        try:
            if path.stat().st_size <= INLINE_MAX_BYTES:
                return {
                    "type": "video",
                    "data": b64(path.read_bytes()),
                    "mime_type": "video/mp4",
                    "processing": processing,
                }
            st = path.stat()
            k = f"{req.cache_key}:{path.resolve()}:{st.st_size}:{int(st.st_mtime)}"
            with _UPLOAD_LOCK:
                hit = _UPLOADS.get(k)
            if hit and not trimmed and time.time() - hit[2] < FILE_TTL_S:
                uri, mime = hit[0], hit[1]
            else:
                uri, mime = self._upload(path)
                if not trimmed:
                    with _UPLOAD_LOCK:
                        _UPLOADS[k] = (uri, mime, time.time())
            return {"type": "video", "uri": uri, "mime_type": mime, "processing": processing}
        finally:
            if trimmed:
                path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ call
    def build_request(self, req: AnalyzeRequest) -> dict[str, Any]:
        parts: list[dict] = []
        if req.history:
            parts.append({"type": "text", "text": history_text(req.history)})
        if req.video is not None:
            parts.append(self._video_part(req))
        for img in limit_images(req.images, self.caps.max_images):
            parts.append({"type": "text", "text": img.label})
            parts.append(
                {
                    "type": "image",
                    "data": b64(fit_jpeg(img.jpeg, self.caps.max_px)),
                    "mime_type": "image/jpeg",
                }
            )
        parts.append({"type": "text", "text": user_text(req)})
        kwargs: dict[str, Any] = {
            "model": req.model or self.model,
            "input": parts,
            "generation_config": {
                "max_output_tokens": req.max_tokens,
                "thinking_level": EFFORT_3[req.effort],
            },
            "store": False,
        }
        if req.system:
            kwargs["system_instruction"] = req.system
        if req.output_model is not None:
            schema = inline_refs(req.output_model.model_json_schema())
            kwargs["response_format"] = {"type": "text", "mime_type": "application/json", "schema": schema}
        return kwargs

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        t0 = now()
        model = req.model or self.model
        try:
            ix = self.client.interactions.create(**self.build_request(req))
        except Exception as exc:  # noqa: BLE001
            return failure(self.key, model, t0, short_error(exc))
        u = getattr(ix, "usage", None)
        cached = getattr(u, "total_cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=max(0, (getattr(u, "total_input_tokens", 0) or 0) - cached),
            # thought tokens are billed as output; assumed reported separately from total_output_tokens
            output_tokens=(getattr(u, "total_output_tokens", 0) or 0)
            + (getattr(u, "total_thought_tokens", 0) or 0),
            cache_read_tokens=cached,
        )
        served = str(getattr(ix, "model", None) or model)
        text = getattr(ix, "output_text", None) or ""
        status = str(getattr(ix, "status", "") or "")
        errors = [str(getattr(e, "message", e)) for e in (getattr(ix, "errors", None) or [])]
        err = None
        if status and status != "completed":
            err = f"Gemini interaction {status}" + (f": {'; '.join(errors)}" if errors else "")
        refused = any("safety" in e.lower() or "blocked" in e.lower() for e in errors)
        parsed = None
        if req.output_model is not None and not refused and text:
            parsed, perr = parse_output(text, req.output_model)
            err = err or perr
        elif req.output_model is not None and err is None:
            err = "Gemini returned no output"
        return AnalyzeResult(
            text=text,
            parsed=parsed,
            usage=usage,
            cost_usd=cost_from_usage(served, usage),
            latency_ms=elapsed_ms(t0),
            model_id=served,
            provider=self.key,
            stop_reason=status or None,
            refused=refused,
            error=err,
            raw=to_jsonable(ix),
        )

    def list_models(self) -> list[str]:
        try:
            names = [str(getattr(m, "name", "")).removeprefix("models/") for m in self.client.models.list()]
        except Exception:  # noqa: BLE001
            return [self.model]
        ids = {n for n in names if n.startswith("gemini")}
        return sorted(ids, key=lambda m: (m != self.model, m)) or [self.model]


@vlms.register(
    "gemini",
    label=PROFILE.label,
    license="Google Gemini API terms (paid tier)",
    requires_modules=("google.genai",),
    requires_keys=(PROFILE.key_field,) if PROFILE.key_field else (),
    notes="Native video with audio via the Interactions API; windows trimmed with ffmpeg.",
    profile="gemini",
)
def make_gemini(**kwargs: Any) -> GeminiVLM:
    return GeminiVLM(**kwargs)
