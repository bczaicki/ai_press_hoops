"""OpenAI-style providers behind VisionLLM, one class per fixed profile (users never type a base URL).

- ``openai``: Responses API. ``responses.parse(text_format=Model)`` for structured output, ``responses.create``
  for free text. Images are ``input_image`` data URLs at ``detail="high"``, each preceded by its label.
- ``openrouter``: Chat Completions on https://openrouter.ai/api/v1. JSON fallback ladder: strict json_schema
  (with ``provider.require_parameters``) -> json_object + schema in the prompt -> plain text + Pydantic parse with
  one repair retry. Video goes as a ``video_url`` data URL, only for Gemini slugs (windows trimmed with ffmpeg).
- ``ollama``: native ``ollama.Client.chat`` with ``format=<JSON schema>``, ``think=False``, temperature 0; at most
  8 images; runs locally, costs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hoops.config import PROFILES, ProviderProfile, get_settings
from hoops.cost import add_usage, cost_from_usage, sum_costs
from hoops.reason.base import AnalyzeRequest, AnalyzeResult, Capabilities, Usage
from hoops.reason.common import (
    CV_HEADER,
    EFFORT_3,
    b64,
    elapsed_ms,
    failure,
    fit_jpeg,
    inline_refs,
    limit_images,
    now,
    parse_output,
    schema_instructions,
    short_error,
    to_jsonable,
    trim_clip,
    user_text,
)
from hoops.registry import vlms

MAX_INLINE_VIDEO_BYTES = 40 * 1024 * 1024
OPENAI_SKIP = (
    "audio",
    "realtime",
    "tts",
    "transcribe",
    "embedding",
    "whisper",
    "dall-e",
    "moderation",
    "search",
    "image",
    "instruct",
    "gpt-3.5",
    "gpt-4-0",
    "gpt-4-1106",
    "gpt-4-32k",
)
OPENAI_TEXT_ONLY = ("gpt-4",)


def _caps(p: ProviderProfile, native_video: bool | None = None) -> Capabilities:
    return Capabilities(
        native_video=p.native_video if native_video is None else native_video,
        max_images=p.max_images,
        max_px=p.max_px,
        json_schema=p.json_schema,
        audio=p.audio,
        local=p.local,
    )


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + b64(jpeg)


def _sorted(ids: list[str], default: str) -> list[str]:
    return sorted(set(ids), key=lambda m: (m != default, m)) or [default]


# ====================================================================== OpenAI (Responses API)


class OpenAIVLM:
    key = "openai"

    def __init__(self, model: str | None = None, *, client: Any = None):
        p = PROFILES["openai"]
        self.model = model or p.default_model
        self.caps = _caps(p)
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=get_settings().key("openai_api_key"))
        self.client = client

    def build_input(self, req: AnalyzeRequest) -> list[dict]:
        frames: list[dict] = []
        for img in limit_images(req.images, self.caps.max_images):
            frames.append({"type": "input_text", "text": img.label})
            url = _data_url(fit_jpeg(img.jpeg, self.caps.max_px))
            frames.append({"type": "input_image", "image_url": url, "detail": "high"})
        if req.cv_context:
            frames.append({"type": "input_text", "text": f"{CV_HEADER}\n{req.cv_context}"})
        items: list[dict] = [{"role": "user", "content": frames}] if frames else []  # stable prefix first
        for q, a in req.history:
            items += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        items.append({"role": "user", "content": [{"type": "input_text", "text": req.prompt}]})
        return items

    def _send(self, kwargs: dict, output_model: Any) -> Any:
        if output_model is not None:
            return self.client.responses.parse(text_format=output_model, **kwargs)
        return self.client.responses.create(**kwargs)

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        t0 = now()
        model = req.model or self.model
        if req.video is not None and not req.images:
            return failure(
                self.key, model, t0, "OpenAI models can't watch video; send frames (contact sheets)."
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "input": self.build_input(req),
            "max_output_tokens": req.max_tokens,
            "reasoning": {"effort": EFFORT_3[req.effort]},
            "store": False,
        }
        if req.system:
            kwargs["instructions"] = req.system
        try:
            try:
                resp = self._send(kwargs, req.output_model)
            except Exception as exc:  # non-reasoning models reject `reasoning`; retry once without it
                if "reasoning" not in str(exc).lower() or getattr(exc, "status_code", None) != 400:
                    raise
                kwargs.pop("reasoning")
                resp = self._send(kwargs, req.output_model)
        except Exception as exc:  # noqa: BLE001
            return failure(self.key, model, t0, short_error(exc))

        u = getattr(resp, "usage", None)
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=max(0, (getattr(u, "input_tokens", 0) or 0) - cached),
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cached,
        )
        served = getattr(resp, "model", None) or model
        text = getattr(resp, "output_text", "") or ""
        status = getattr(resp, "status", None)
        common = {
            "usage": usage,
            "cost_usd": cost_from_usage(served, usage),
            "latency_ms": elapsed_ms(t0),
            "model_id": served,
            "provider": self.key,
            "stop_reason": status,
            "raw": to_jsonable(resp),
        }
        refusal = _refusal(resp)
        if refusal:
            return AnalyzeResult(
                text=text, parsed=None, refused=True, error=f"OpenAI declined: {refusal}", **common
            )
        parsed, err = None, None
        if req.output_model is not None:
            parsed = getattr(resp, "output_parsed", None)
            if not isinstance(parsed, req.output_model):
                parsed, err = parse_output(text, req.output_model)
        if status == "incomplete" and err is None and parsed is None:
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            err = f"response incomplete ({reason})"
        return AnalyzeResult(text=text, parsed=parsed, error=err, **common)

    def list_models(self) -> list[str]:
        try:
            ids = [m.id for m in self.client.models.list()]
        except Exception:  # noqa: BLE001
            return [self.model]
        keep = [
            i
            for i in ids
            if i.startswith(("gpt-", "o3", "o4"))
            and i not in OPENAI_TEXT_ONLY
            and not any(s in i for s in OPENAI_SKIP)
        ]
        return _sorted(keep, self.model)


def _refusal(resp: Any) -> str | None:
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", None) or "refused"
    return None


# ====================================================================== OpenRouter (Chat Completions)


class OpenRouterVLM:
    key = "openrouter"

    def __init__(self, model: str | None = None, *, client: Any = None):
        p = PROFILES["openrouter"]
        self.profile = p
        self.model = model or p.default_model
        self.caps = _caps(p, native_video=self.video_ok(self.model))
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=p.base_url,
                api_key=get_settings().key("openrouter_api_key"),
                default_headers={"X-Title": "ai_press_hoops"},
            )
        self.client = client

    @staticmethod
    def video_ok(model: str) -> bool:
        return model.lstrip("~").startswith("google/gemini")

    def _video_part(self, req: AnalyzeRequest) -> dict | None:
        v = req.video
        if v is None:
            return None
        path, tmp = Path(v.path), None
        if v.t0_s is not None or v.t1_s is not None:
            start = v.t0_s or 0.0
            tmp = path = trim_clip(path, start, v.t1_s if v.t1_s is not None else start + 1e6)
        try:
            data = path.read_bytes()
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        if len(data) > MAX_INLINE_VIDEO_BYTES:
            raise ValueError(
                f"clip is {len(data) / 1e6:.0f} MB; inline video is capped at 40 MB (send a window)"
            )
        return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64(data)}}

    def build_messages(self, req: AnalyzeRequest, model: str, *, schema_hint: bool) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": req.system}] if req.system else []
        for q, a in req.history:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        content: list[dict] = [
            {"type": "text", "text": user_text(req, schema_hint=schema_hint)}
        ]  # text first
        for img in limit_images(req.images, self.caps.max_images):
            content.append({"type": "text", "text": img.label})
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(fit_jpeg(img.jpeg, self.caps.max_px))}}
            )
        if req.video is not None and self.video_ok(model):
            content.append(self._video_part(req))
        msgs.append({"role": "user", "content": content})
        return msgs

    def _call(self, model: str, messages: list[dict], max_tokens: int, **kw: Any) -> Any:
        extra = {"usage": {"include": True}}
        if kw.pop("require", False):
            extra["provider"] = {"require_parameters": True}
        return self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, extra_body=extra, **kw
        )

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        t0 = now()
        model = req.model or self.model
        if req.video is not None and not req.images and not self.video_ok(model):
            return failure(self.key, model, t0, f"{model} can't take video on OpenRouter; send frames.")
        out_model = req.output_model
        calls: list[Any] = []
        notes: list[str] = []

        def finish(
            text: str, parsed: Any, err: str | None, mode: str, refusal: str | None = None
        ) -> AnalyzeResult:
            usage, costs = Usage(), []
            for r in calls:
                u, c = _chat_usage(r, model)
                usage, costs = add_usage(usage, u), [*costs, c]
            last = calls[-1] if calls else None
            served = getattr(last, "model", None) or model
            return AnalyzeResult(
                text=text,
                parsed=parsed,
                usage=usage,
                cost_usd=sum_costs(costs),
                latency_ms=elapsed_ms(t0),
                model_id=served,
                provider=self.key,
                stop_reason=_finish_reason(last),
                refused=refusal is not None,
                error=f"model declined: {refusal}" if refusal else err,
                raw={"mode": mode, "notes": notes, "responses": [to_jsonable(r) for r in calls]},
            )

        try:
            if out_model is None:
                calls.append(
                    self._call(model, self.build_messages(req, model, schema_hint=False), req.max_tokens)
                )
                text, refusal = _chat_text(calls[-1])
                return finish(text, None, None, "text", refusal)

            schema = inline_refs(out_model.model_json_schema())
            rungs: list[tuple[str, dict, bool]] = []
            if self.profile.json_schema != "none":
                rf = {
                    "type": "json_schema",
                    "json_schema": {"name": out_model.__name__, "strict": True, "schema": schema},
                }
                rungs.append(("json_schema", {"response_format": rf, "require": True}, False))
                rungs.append(
                    ("json_object", {"response_format": {"type": "json_object"}, "require": True}, True)
                )
            rungs.append(("text", {}, True))
            for mode, kw, hint in rungs:
                messages = self.build_messages(req, model, schema_hint=hint)
                try:
                    resp = self._call(model, messages, req.max_tokens, **kw)
                except Exception as exc:  # unsupported parameters -> next rung; anything else is fatal
                    if getattr(exc, "status_code", None) in (400, 404, 422) and mode != "text":
                        notes.append(f"{mode}: {short_error(exc)}")
                        continue
                    raise
                calls.append(resp)
                text, refusal = _chat_text(resp)
                if refusal:
                    return finish(text, None, None, mode, refusal)
                parsed, err = parse_output(text, out_model)
                if parsed is not None:
                    return finish(text, parsed, None, mode)
                notes.append(f"{mode}: {err}")
                repair = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"That reply didn't validate ({err}). {schema_instructions(out_model)}",
                    },
                ]
                calls.append(self._call(model, repair, req.max_tokens))
                text, refusal = _chat_text(calls[-1])
                parsed, err = parse_output(text, out_model)
                return finish(text, parsed, err, f"{mode}+repair", refusal)
            return finish("", None, "no structured-output mode worked", "none")
        except Exception as exc:  # noqa: BLE001
            if calls:
                return finish("", None, short_error(exc), "error")
            return failure(self.key, model, t0, short_error(exc), raw={"notes": notes})

    def list_models(self) -> list[str]:
        try:
            models = list(self.client.models.list())
        except Exception:  # noqa: BLE001
            return [self.model]
        ids = []
        for m in models:
            arch = (getattr(m, "model_extra", None) or {}).get("architecture") or {}
            if "image" in (arch.get("input_modalities") or []):
                ids.append(m.id)
        return _sorted(ids, self.model)


def _chat_text(resp: Any) -> tuple[str, str | None]:
    choice = (getattr(resp, "choices", None) or [None])[0]
    msg = getattr(choice, "message", None)
    return (getattr(msg, "content", None) or ""), getattr(msg, "refusal", None)


def _finish_reason(resp: Any) -> str | None:
    choice = (getattr(resp, "choices", None) or [None])[0]
    return getattr(choice, "finish_reason", None)


def _chat_usage(resp: Any, model: str) -> tuple[Usage, float | None]:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage(), None
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    usage = Usage(
        input_tokens=max(0, (getattr(u, "prompt_tokens", 0) or 0) - cached),
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_tokens=cached,
    )
    reported = (getattr(u, "model_extra", None) or {}).get("cost")  # OpenRouter's own USD figure
    return usage, float(reported) if reported is not None else cost_from_usage(model, usage)


# ====================================================================== Ollama (local, native client)


# Installed-model preference when the configured default isn't pulled (vision-capable families first).
OLLAMA_VISION_HINTS = (
    "qwen3.5",
    "qwen3-vl",
    "qwen2.5vl",
    "gemma4",
    "gemma3",
    "llama3.2-vision",
    "minicpm-v",
    "llava",
    "moondream",
)


class OllamaVLM:
    key = "ollama"

    def __init__(self, model: str | None = None, *, client: Any = None, host: str | None = None):
        p = PROFILES["ollama"]
        self._explicit = model is not None
        self.model = model or get_settings().ollama_model or p.default_model
        self._installed: list[str] | None = None
        self.caps = _caps(p)
        self.host = host or get_settings().ollama_host
        if client is None:
            import ollama

            client = ollama.Client(host=self.host)
        self.client = client

    def build_messages(self, req: AnalyzeRequest) -> list[dict]:
        images = limit_images(req.images, self.caps.max_images)
        text = user_text(req, schema_hint=req.output_model is not None)
        if images:
            listing = "\n".join(f"Image {i + 1}: {img.label}" for i, img in enumerate(images))
            text = f"The {len(images)} attached images, in order:\n{listing}\n\n{text}"
        msgs: list[dict] = [{"role": "system", "content": req.system}] if req.system else []
        for q, a in req.history:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msg: dict[str, Any] = {"role": "user", "content": text}
        if images:
            msg["images"] = [fit_jpeg(i.jpeg, self.caps.max_px) for i in images]
        msgs.append(msg)
        return msgs

    def resolve_model(self, wanted: str, explicit: bool) -> str:
        """``wanted`` if installed; otherwise (unless explicitly requested) the best installed vision model."""
        if self._installed is None:
            try:
                self._installed = [m.model for m in self.client.list().models if m.model]
            except Exception:  # noqa: BLE001 - unreachable server is reported by chat()
                self._installed = []
        names = self._installed
        if not names or explicit or wanted in names or f"{wanted}:latest" in names:
            return wanted
        for hint in OLLAMA_VISION_HINTS:
            for name in names:
                if name.startswith(hint):
                    return name
        return wanted

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        t0 = now()
        model = self.resolve_model(req.model or self.model, explicit=bool(req.model) or self._explicit)
        if req.video is not None and not req.images:
            return failure(
                self.key, model, t0, "Local models can't watch video; send frames (contact sheets)."
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self.build_messages(req),
            "think": False,
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": min(req.max_tokens, 4096)},
        }
        if req.output_model is not None:
            kwargs["format"] = inline_refs(req.output_model.model_json_schema())
        try:
            try:
                resp = self.client.chat(**kwargs)
            except Exception as exc:  # models without a thinking switch reject `think`
                if "think" not in str(exc).lower():
                    raise
                kwargs.pop("think")
                resp = self.client.chat(**kwargs)
        except ConnectionError as exc:
            return failure(
                self.key, model, t0, f"Ollama isn't reachable at {self.host}; start the Ollama app ({exc})."
            )
        except Exception as exc:  # noqa: BLE001
            msg = short_error(exc)
            if "not found" in msg.lower():
                msg = f"Model '{model}' isn't installed locally: run `ollama pull {model}`."
            return failure(self.key, model, t0, msg)
        message = getattr(resp, "message", None)
        text = getattr(message, "content", None) or ""
        usage = Usage(
            input_tokens=getattr(resp, "prompt_eval_count", 0) or 0,
            output_tokens=getattr(resp, "eval_count", 0) or 0,
        )
        parsed, err = parse_output(text, req.output_model) if req.output_model is not None else (None, None)
        return AnalyzeResult(
            text=text,
            parsed=parsed,
            usage=usage,
            cost_usd=0.0,
            latency_ms=elapsed_ms(t0),
            model_id=getattr(resp, "model", None) or model,
            provider=self.key,
            stop_reason=getattr(resp, "done_reason", None),
            error=err,
            raw=to_jsonable(resp),
        )

    def list_models(self) -> list[str]:
        try:
            ids = [m.model for m in self.client.list().models if m.model]
        except Exception:  # noqa: BLE001
            return [self.model]
        return _sorted(ids, self.model)


# ====================================================================== registry


def _register(profile_key: str, cls: type, module: str, notes: str) -> None:
    p = PROFILES[profile_key]

    @vlms.register(
        profile_key,
        label=p.label,
        license="local (model license applies)" if p.local else f"{p.sends_data_to} API terms",
        requires_modules=(module,),
        requires_keys=(p.key_field,) if p.key_field else (),
        notes=notes,
        profile=profile_key,
    )
    def factory(**kwargs: Any) -> Any:
        return cls(**kwargs)


_register("openai", OpenAIVLM, "openai", "Responses API; frames as images; strict structured output.")
_register(
    "openrouter", OpenRouterVLM, "openai", "Any model slug; JSON fallback ladder; video for Gemini slugs."
)
_register("ollama", OllamaVLM, "ollama", "Runs on this Mac; at most 8 images; free.")

__all__ = ["OllamaVLM", "OpenAIVLM", "OpenRouterVLM"]
