from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from hoops import registry
from hoops.config import get_settings
from hoops.reason import gemini_vlm
from hoops.reason.base import AnalyzeRequest, ImagePart, VideoPart
from hoops.reason.gemini_vlm import GeminiVLM
from hoops.schemas import ClipSummary, VLMResult
from tests.helpers import live, requires_ffmpeg

SUMMARY = {
    "headline": "White wins the run",
    "narrative": "A quick 4-0 run.",
    "key_moments": [],
    "caveats": [],
}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (200, 100, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def _ix(text: str, status: str = "completed", errors=None):
    usage = SimpleNamespace(
        total_input_tokens=6000, total_output_tokens=400, total_thought_tokens=600, total_cached_tokens=1000
    )
    return SimpleNamespace(
        output_text=text, status=status, errors=errors, usage=usage, model="gemini-3.8-flash"
    )


class FakeGenai:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.uploads: list[str] = []
        self.gets = 0

        def create(**kw):
            self.calls.append(kw)
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def upload(file):
            self.uploads.append(file)
            return SimpleNamespace(
                name="files/abc",
                uri="https://files/abc",
                mime_type="video/mp4",
                state=SimpleNamespace(name="PROCESSING"),
            )

        def get(name):
            self.gets += 1
            return SimpleNamespace(
                name=name,
                uri="https://files/abc",
                mime_type="video/mp4",
                state=SimpleNamespace(name="ACTIVE"),
            )

        self.interactions = SimpleNamespace(create=create)
        self.files = SimpleNamespace(upload=upload, get=get)
        self.models = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name="models/gemini-3.5-flash-lite"),
                SimpleNamespace(name="models/gemini-3.8-flash"),
                SimpleNamespace(name="models/imagen-4"),
            ]
        )


@requires_ffmpeg
def test_native_video_window_is_trimmed_and_inlined(testsrc_video):
    client = FakeGenai([_ix(json.dumps(SUMMARY))])
    req = AnalyzeRequest(
        prompt="Summarize.",
        system="Be careful.",
        video=VideoPart(path=testsrc_video, t0_s=0.5, t1_s=1.5, fps=8.0),
        cv_context='{"shots": 1}',
        output_model=ClipSummary,
        effort="low",
    )
    res = GeminiVLM(client=client).analyze(req)
    assert res.error is None and isinstance(res.parsed, ClipSummary)
    kw = client.calls[0]
    video = kw["input"][0]
    assert video["type"] == "video" and video["mime_type"] == "video/mp4"
    assert video["processing"] == {"type": "static", "fps": 8.0}  # no clip offsets: the window was trimmed
    clip = base64.b64decode(video["data"])
    assert 0 < len(clip) < len(testsrc_video.read_bytes())
    assert kw["input"][-1]["text"].startswith("CV context") and kw["input"][-1]["text"].endswith("Summarize.")
    assert kw["response_format"]["mime_type"] == "application/json" and "schema" in kw["response_format"]
    assert kw["generation_config"] == {"max_output_tokens": 16000, "thinking_level": "low"}
    assert kw["system_instruction"] == "Be careful." and kw["store"] is False
    assert (res.usage.input_tokens, res.usage.cache_read_tokens, res.usage.output_tokens) == (
        5000,
        1000,
        1000,
    )
    assert res.cost_usd == pytest.approx((5000 * 0.75 + 1000 * 0.75 * 0.1 + 1000 * 3.75) / 1e6)


@requires_ffmpeg
def test_large_video_uploads_once_and_polls(testsrc_video, monkeypatch):
    monkeypatch.setattr(gemini_vlm, "INLINE_MAX_BYTES", 0)
    monkeypatch.setattr(gemini_vlm, "_UPLOADS", {})
    client = FakeGenai([_ix("ok"), _ix("ok again")])
    vlm = GeminiVLM(client=client, poll_s=0)
    req = AnalyzeRequest(
        prompt="What happens?", video=VideoPart(path=testsrc_video, fps=1.0), cache_key="run-9"
    )
    assert vlm.analyze(req).text == "ok"
    assert vlm.analyze(req).text == "ok again"
    assert len(client.uploads) == 1 and client.gets == 1
    assert client.calls[1]["input"][0]["uri"] == "https://files/abc"


def test_images_are_labeled_and_schema_has_no_refs():
    client = FakeGenai([_ix(json.dumps({"events": [], "narrative": "n", "unsure_about": []}))])
    req = AnalyzeRequest(
        prompt="List events.",
        images=[ImagePart(jpeg=_jpeg(), label="Sheet S1: tiles S1#1..S1#9")],
        output_model=VLMResult,
        history=[("Who scored first?", "White #4.")],
    )
    res = GeminiVLM(client=client).analyze(req)
    assert res.error is None and isinstance(res.parsed, VLMResult)
    parts = client.calls[0]["input"]
    assert "Earlier question: Who scored first?" in parts[0]["text"]
    assert parts[1] == {"type": "text", "text": "Sheet S1: tiles S1#1..S1#9"}
    assert parts[2]["type"] == "image" and parts[2]["mime_type"] == "image/jpeg"
    assert "$ref" not in json.dumps(client.calls[0]["response_format"])


def test_errors_and_incomplete_status():
    res = GeminiVLM(client=FakeGenai([RuntimeError("quota")])).analyze(AnalyzeRequest(prompt="hi"))
    assert res.error and "quota" in res.error
    res = GeminiVLM(
        client=FakeGenai([_ix("", status="failed", errors=[SimpleNamespace(message="blocked by safety")])])
    ).analyze(AnalyzeRequest(prompt="hi", output_model=ClipSummary))
    assert res.refused and "failed" in res.error and res.parsed is None


def test_list_models_and_registry():
    assert GeminiVLM(client=FakeGenai([])).list_models() == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    entry = registry.vlms.entry("gemini")
    assert entry.requires_keys == ("gemini_api_key",) and entry.requires_modules == ("google.genai",)


@live
@requires_ffmpeg
def test_live_gemini_native_video(testsrc_video):
    if not get_settings().key("gemini_api_key"):
        pytest.skip("no GEMINI_API_KEY")
    res = GeminiVLM().analyze(
        AnalyzeRequest(
            prompt="Describe this test pattern video in one sentence.",
            video=VideoPart(path=testsrc_video, t0_s=0.0, t1_s=1.0, fps=2.0),
            effort="low",
            max_tokens=2000,
        )
    )
    print("gemini:", res.model_id, res.usage, res.cost_usd, res.latency_ms, repr(res.text[:200]))
    assert res.error is None and res.text
