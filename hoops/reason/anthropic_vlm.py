"""Claude via the official ``anthropic`` SDK.

- Requests go through ``client.beta.messages`` so server-side refusal fallbacks (``fallbacks="default"`` +
  beta ``server-side-fallback-2026-07-01``) are on by default for the Opus/Sonnet 5.x family.
- Opus 5.5 always thinks: never send ``thinking``, ``temperature``/``top_p``, an assistant prefill, or a forced
  ``tool_choice``. Depth is set with ``output_config.effort`` (always explicit).
- Structured output uses ``output_config.format`` (schema from ``anthropic.transform_schema``) and is validated
  here, not by ``messages.parse()``: parse() raises on an invalid or truncated reply and loses the (billed)
  response, while this path keeps usage, cost and the raw reply for every outcome.
- Images are base64 blocks, each preceded by its text label. With ``cache_key`` set, frames are uploaded once with
  the Files API and the frames prefix gets ``cache_control`` so Q&A follow-ups hit the prompt cache.
- Claude won't identify people by face; prompts ask for jersey color/number only.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from hoops.config import PROFILES, get_settings
from hoops.cost import cost_from_usage
from hoops.reason.base import AnalyzeRequest, AnalyzeResult, Capabilities, ImagePart, Usage
from hoops.reason.common import (
    CV_HEADER,
    b64,
    elapsed_ms,
    failure,
    fit_jpeg,
    limit_images,
    now,
    parse_output,
    short_error,
    to_jsonable,
)
from hoops.registry import vlms

PROFILE = PROFILES["claude"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = (
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-fable-5",
)
NO_EFFORT_PREFIXES = ("claude-haiku-4", "claude-sonnet-4-5", "claude-3")
MANY_IMAGES = 20  # above this, every image must be <= 2000 px
MANY_IMAGES_MAX_PX = 2000
MAX_IMAGES = 100
MAX_REQUEST_BYTES = 20 * 1024 * 1024  # raw JPEG bytes: x4/3 for base64 + JSON stays under the 32 MB cap
STREAM_ABOVE_TOKENS = 16_000

_FILE_IDS: dict[
    str, str
] = {}  # "<cache_key>:<sha256>" -> Files API id; process-wide so new adapters reuse uploads
_FILE_LOCK = threading.Lock()


class ClaudeVLM:
    key = "claude"

    def __init__(self, model: str | None = None, *, client: Any = None, fallbacks: bool = True):
        self.model = model or PROFILE.default_model
        self.caps = Capabilities(
            native_video=False, max_images=PROFILE.max_images, max_px=PROFILE.max_px, json_schema="strict"
        )
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=get_settings().key("anthropic_api_key"))
        self.client = client
        self.fallbacks = fallbacks

    # ------------------------------------------------------------------ request
    def _image_source(self, img: ImagePart, jpeg: bytes, cache_key: str | None) -> dict:
        if cache_key:
            k = f"{cache_key}:{hashlib.sha256(jpeg).hexdigest()}"
            with _FILE_LOCK:
                file_id = _FILE_IDS.get(k)
            if file_id is None:
                try:
                    name = f"{(img.tile_id or 'sheet').replace('#', '_')}.jpg"
                    file_id = self.client.files.upload(file=(name, jpeg, "image/jpeg")).id
                    with _FILE_LOCK:
                        _FILE_IDS[k] = file_id
                except Exception:  # noqa: BLE001 - inline base64 still works
                    file_id = None
            if file_id:
                return {"type": "file", "file_id": file_id}
        return {"type": "base64", "media_type": "image/jpeg", "data": b64(jpeg)}

    def build_request(self, req: AnalyzeRequest) -> dict[str, Any]:
        model = req.model or self.model
        images = limit_images(req.images, MAX_IMAGES)
        max_px = self.caps.max_px if len(images) <= MANY_IMAGES else MANY_IMAGES_MAX_PX
        jpegs = [fit_jpeg(i.jpeg, max_px) for i in images]
        if sum(map(len, jpegs)) > MAX_REQUEST_BYTES:
            jpegs = [fit_jpeg(j, 1280) for j in jpegs]

        content: list[dict] = []
        for img, jpeg in zip(images, jpegs, strict=True):
            content.append({"type": "text", "text": img.label})
            content.append({"type": "image", "source": self._image_source(img, jpeg, req.cache_key)})
        if req.cv_context:
            content.append({"type": "text", "text": f"{CV_HEADER}\n{req.cv_context}"})
        cache = {"type": "ephemeral"} if req.cache_key else None
        if cache and content:
            content[-1]["cache_control"] = cache  # stable frames + context prefix

        messages: list[dict] = [{"role": "user", "content": content}] if content else []
        for i, (q, a) in enumerate(req.history):
            messages.append({"role": "user", "content": q})
            block: dict[str, Any] = {"type": "text", "text": a}
            if cache and i == len(req.history) - 1:
                block["cache_control"] = cache  # the conversation so far is also a stable prefix
            messages.append({"role": "assistant", "content": [block]})
        messages.append({"role": "user", "content": req.prompt})

        kwargs: dict[str, Any] = {"model": model, "max_tokens": req.max_tokens, "messages": messages}
        if req.system:
            kwargs["system"] = [{"type": "text", "text": req.system}]
        output_config: dict[str, Any] = {}
        if not model.startswith(NO_EFFORT_PREFIXES):
            output_config["effort"] = req.effort
        if req.output_model is not None:
            import anthropic

            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(req.output_model),
            }
        if output_config:
            kwargs["output_config"] = output_config
        if self.fallbacks and model in FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    # ------------------------------------------------------------------ call
    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        t0 = now()
        model = req.model or self.model
        if req.video is not None and not req.images:
            return failure(
                self.key, model, t0, "Claude can't watch video directly; send frames (contact sheets)."
            )
        try:
            kwargs = self.build_request(req)
            if req.max_tokens > STREAM_ABOVE_TOKENS:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                msg = self.client.beta.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - adapters report provider errors, never raise
            return failure(self.key, model, t0, short_error(exc))
        return self._result(req, msg, model, t0)

    def _result(self, req: AnalyzeRequest, msg: Any, model: str, t0: float) -> AnalyzeResult:
        u = msg.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        served = getattr(msg, "model", None) or model  # differs from `model` when a fallback answered
        cost = cost_from_usage(served, usage)
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
        stop = getattr(msg, "stop_reason", None)
        common = {
            "usage": usage,
            "cost_usd": cost,
            "latency_ms": elapsed_ms(t0),
            "model_id": served,
            "provider": self.key,
            "stop_reason": stop,
            "raw": to_jsonable(msg),
        }
        if stop == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            return AnalyzeResult(
                text=text,
                parsed=None,
                refused=True,
                error=f"Claude declined this request (category: {category}).",
                **common,
            )
        parsed, err = None, None
        if req.output_model is not None:
            parsed, err = parse_output(text, req.output_model)
            if err and stop == "max_tokens":
                err = f"reply was cut off at max_tokens ({req.max_tokens}); {err}"
        return AnalyzeResult(text=text, parsed=parsed, error=err, **common)

    def list_models(self) -> list[str]:
        try:
            ids = [m.id for m in self.client.models.list()]
        except Exception:  # noqa: BLE001
            return [self.model]
        return sorted(ids, key=lambda m: (m != self.model, m)) or [self.model]


@vlms.register(
    "claude",
    label=PROFILE.label,
    license="Anthropic commercial terms",
    requires_modules=("anthropic",),
    requires_keys=(PROFILE.key_field,) if PROFILE.key_field else (),
    notes="Frames as contact sheets; strict structured output; refusal fallbacks on.",
    profile="claude",
)
def make_claude(**kwargs: Any) -> ClaudeVLM:
    return ClaudeVLM(**kwargs)
