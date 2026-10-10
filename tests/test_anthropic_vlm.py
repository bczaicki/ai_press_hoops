from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from hoops import registry
from hoops.reason import anthropic_vlm
from hoops.reason.anthropic_vlm import FALLBACK_BETA, ClaudeVLM
from hoops.reason.base import AnalyzeRequest, ImagePart, VideoPart
from hoops.schemas import VM, WindowVerdict
from tests.helpers import live

VERDICT = {
    "verdict": "made",
    "shot_kind": "jumper",
    "confidence": 0.9,
    "evidence_tiles": ["R1#5"],
    "shooter_description": "#23 in white",
    "reason": "Ball drops through the net in R1#5.",
}


def _jpeg(w: int = 64, h: int = 64, color=(255, 140, 0)) -> bytes:
    img = Image.new("RGB", (w, h), (0, 0, 0))
    ImageDraw.Draw(img).ellipse((w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _msg(text: str, stop: str = "end_turn", model: str = "claude-opus-5-5", **usage) -> SimpleNamespace:
    u = SimpleNamespace(
        input_tokens=usage.get("input_tokens", 1000),
        output_tokens=usage.get("output_tokens", 200),
        cache_read_input_tokens=usage.get("cache_read", 0),
        cache_creation_input_tokens=usage.get("cache_write", 0),
    )
    msg = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
        usage=u,
        model=model,
    )
    msg.to_dict = lambda: {"model": model, "stop_reason": stop, "content": [{"type": "text", "text": text}]}
    return msg


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, reply=None, error: Exception | None = None):
        self.calls: list[dict] = []
        self.streams: list[dict] = []
        self.uploads: list[tuple] = []
        self.reply = reply if reply is not None else _msg(json.dumps(VERDICT))
        self.error = error

        def create(**kw):
            self.calls.append(kw)
            if self.error:
                raise self.error
            return self.reply

        def stream(**kw):
            self.streams.append(kw)
            return FakeStream(self.reply)

        def upload(file):
            self.uploads.append(file)
            return SimpleNamespace(id=f"file_{len(self.uploads)}")

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
        self.files = SimpleNamespace(upload=upload)
        self.models = SimpleNamespace(
            list=lambda: [SimpleNamespace(id="claude-haiku-5-5"), SimpleNamespace(id="claude-opus-5-5")]
        )


def _req(**kw) -> AnalyzeRequest:
    base = {
        "prompt": "Was the shot made?",
        "system": "You are a careful analyst.",
        "images": [ImagePart(jpeg=_jpeg(), label="Sheet R1: tiles R1#1..R1#8", t_s=1.0)],
        "cv_context": '{"rim": "visible"}',
        "output_model": WindowVerdict,
        "effort": "low",
        "max_tokens": 8000,
        "purpose": "adjudicate",
    }
    base.update(kw)
    return AnalyzeRequest(**base)


def test_request_shape_follows_opus_rules():
    client = FakeClient()
    res = ClaudeVLM(client=client).analyze(_req())
    assert res.error is None, res.error
    kw = client.calls[0]
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "output_format"):
        assert banned not in kw
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["effort"] == "low"
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]
    assert kw["system"] == [{"type": "text", "text": "You are a careful analyst."}]
    msgs = kw["messages"]
    assert msgs[-1] == {"role": "user", "content": "Was the shot made?"}  # no assistant prefill
    first = msgs[0]["content"]
    assert first[0] == {"type": "text", "text": "Sheet R1: tiles R1#1..R1#8"}  # label precedes the image
    assert first[1]["type"] == "image" and first[1]["source"]["type"] == "base64"
    assert first[2]["text"].startswith("CV context")
    assert "cache_control" not in first[-1]  # no cache_key, no cache writes


def test_parsed_result_usage_and_cost():
    res = ClaudeVLM(client=FakeClient()).analyze(_req())
    assert isinstance(res.parsed, WindowVerdict) and res.parsed.verdict == "made"
    assert res.usage.input_tokens == 1000 and res.usage.output_tokens == 200
    assert res.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    assert res.provider == "claude" and res.model_id == "claude-opus-5-5"
    json.dumps(res.raw)  # JSON-serializable


def test_free_text_has_no_format_but_keeps_effort():
    client = FakeClient(reply=_msg("A quick recap."))
    res = ClaudeVLM(client=client).analyze(_req(output_model=None, effort="medium"))
    assert res.text == "A quick recap." and res.parsed is None and res.error is None
    assert client.calls[0]["output_config"] == {"effort": "medium"}


def test_refusal_sets_refused():
    res = ClaudeVLM(client=FakeClient(reply=_msg("", stop="refusal"))).analyze(_req())
    assert res.refused and res.parsed is None and "cyber" in res.error
    assert res.cost_usd is not None


def test_invalid_json_keeps_usage_and_reports_error():
    res = ClaudeVLM(client=FakeClient(reply=_msg('{"verdict": "made"', stop="max_tokens"))).analyze(_req())
    assert res.parsed is None and "max_tokens" in res.error
    assert res.usage.output_tokens == 200


def test_provider_errors_never_raise():
    res = ClaudeVLM(client=FakeClient(error=RuntimeError("boom"))).analyze(_req())
    assert res.error and "boom" in res.error and res.parsed is None


def test_video_without_frames_is_rejected():
    res = ClaudeVLM(client=FakeClient()).analyze(_req(images=[], video=VideoPart(path="x.mp4")))
    assert res.error and "frames" in res.error


def test_streams_large_outputs():
    client = FakeClient()
    ClaudeVLM(client=client).analyze(_req(max_tokens=64000))
    assert not client.calls and client.streams[0]["max_tokens"] == 64000


def test_files_api_reuse_and_prompt_cache(monkeypatch):
    monkeypatch.setattr(anthropic_vlm, "_FILE_IDS", {})
    client = FakeClient()
    vlm = ClaudeVLM(client=client)
    req = _req(cache_key="run-1", history=[("How many shots?", "Three.")])
    vlm.analyze(req)
    vlm.analyze(req)
    assert len(client.uploads) == 1  # uploaded once, reused on the follow-up
    msgs = client.calls[1]["messages"]
    frames = msgs[0]["content"]
    assert frames[1]["source"] == {"type": "file", "file_id": "file_1"}
    assert frames[-1]["cache_control"] == {"type": "ephemeral"}
    assert msgs[1] == {"role": "user", "content": "How many shots?"}
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_fallbacks_only_for_supported_models():
    client = FakeClient()
    ClaudeVLM(client=client).analyze(_req(model="claude-haiku-5-5"))
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_list_models_puts_default_first():
    assert ClaudeVLM(client=FakeClient()).list_models() == ["claude-opus-5-5", "claude-haiku-5-5"]


def test_registered_under_profile_key():
    entry = registry.vlms.entry("claude")
    assert entry.requires_keys == ("anthropic_api_key",) and entry.requires_modules == ("anthropic",)


class Shape(VM):
    color: str
    shape: str


@live
def test_live_claude_parse_and_text():
    vlm = ClaudeVLM()
    img = [ImagePart(jpeg=_jpeg(), label="Frame A t=0.00s")]
    res = vlm.analyze(
        AnalyzeRequest(
            prompt="Name the color and shape of the object in Frame A.",
            images=img,
            output_model=Shape,
            effort="low",
            max_tokens=2000,
        )
    )
    print("claude parse:", res.model_id, res.usage, res.cost_usd, res.latency_ms, res.parsed, res.stop_reason)
    assert res.error is None, res.error
    assert isinstance(res.parsed, Shape) and "orange" in res.parsed.color.lower()
    res2 = vlm.analyze(
        AnalyzeRequest(
            prompt="In one word, what color is the circle?", images=img, effort="low", max_tokens=2000
        )
    )
    print("claude text:", res2.model_id, res2.usage, res2.cost_usd, repr(res2.text))
    assert res2.error is None and "orange" in res2.text.lower()
    assert (res.cost_usd or 0) + (res2.cost_usd or 0) < 0.05
