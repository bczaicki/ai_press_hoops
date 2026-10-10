from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from hoops import registry
from hoops.reason.base import AnalyzeRequest, ImagePart, VideoPart
from hoops.reason.openai_compat import OllamaVLM, OpenAIVLM, OpenRouterVLM
from hoops.schemas import VM, WindowVerdict
from tests.helpers import live, requires_ffmpeg

VERDICT = {
    "verdict": "missed",
    "shot_kind": "layup",
    "confidence": 0.7,
    "evidence_tiles": ["R1#3"],
    "shooter_description": None,
    "reason": "Rolls off the rim in R1#3.",
}


def _jpeg(w: int = 64, h: int = 64) -> bytes:
    img = Image.new("RGB", (w, h), (0, 0, 0))
    ImageDraw.Draw(img).ellipse((w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=(255, 140, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _req(**kw) -> AnalyzeRequest:
    base = {
        "prompt": "Was the shot made?",
        "system": "Be careful.",
        "images": [ImagePart(jpeg=_jpeg(), label="Sheet R1: tiles R1#1..R1#8")],
        "cv_context": '{"rim": "visible"}',
        "output_model": WindowVerdict,
        "effort": "low",
        "max_tokens": 4000,
    }
    base.update(kw)
    return AnalyzeRequest(**base)


class HTTPError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


# ----------------------------------------------------------------- OpenAI


def _response(text: str, parsed=None, refusal: str | None = None, status: str = "completed"):
    content = (
        [SimpleNamespace(type="refusal", refusal=refusal)]
        if refusal
        else [SimpleNamespace(type="output_text")]
    )
    return SimpleNamespace(
        output_text=text,
        output_parsed=parsed,
        output=[SimpleNamespace(type="message", content=content)],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=200)
        ),
        model="gpt-6.1-sol",
        status=status,
        incomplete_details=None,
    )


class FakeOpenAI:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

        def make(kind):
            def fn(**kw):
                self.calls.append((kind, kw))
                r = self.replies.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

            return fn

        self.responses = SimpleNamespace(parse=make("parse"), create=make("create"))
        self.models = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(id=i) for i in ("gpt-6-luna", "gpt-6.1-sol", "whisper-1", "gpt-6-tts")
            ]
        )


def test_openai_structured_request_and_usage():
    parsed = WindowVerdict(**VERDICT)
    client = FakeOpenAI([_response(json.dumps(VERDICT), parsed)])
    res = OpenAIVLM(client=client).analyze(_req())
    assert res.error is None and res.parsed == parsed
    kind, kw = client.calls[0]
    assert kind == "parse" and kw["text_format"] is WindowVerdict
    assert (
        kw["reasoning"] == {"effort": "low"} and kw["instructions"] == "Be careful." and kw["store"] is False
    )
    frames = kw["input"][0]["content"]
    assert frames[0] == {"type": "input_text", "text": "Sheet R1: tiles R1#1..R1#8"}
    assert frames[1]["type"] == "input_image" and frames[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert frames[1]["detail"] == "high" and frames[2]["text"].startswith("CV context")
    assert kw["input"][-1]["content"][0]["text"] == "Was the shot made?"
    assert (res.usage.input_tokens, res.usage.cache_read_tokens, res.usage.output_tokens) == (800, 200, 100)
    assert res.cost_usd == pytest.approx((800 * 2 + 200 * 2 * 0.05 + 100 * 10) / 1e6)


def test_openai_retries_without_reasoning_and_free_text():
    client = FakeOpenAI([HTTPError("Unsupported parameter: 'reasoning'", 400), _response("Orange.")])
    res = OpenAIVLM(client=client).analyze(_req(output_model=None))
    assert res.text == "Orange." and res.error is None
    assert [c[0] for c in client.calls] == ["create", "create"] and "reasoning" not in client.calls[1][1]


def test_openai_refusal_and_errors():
    res = OpenAIVLM(client=FakeOpenAI([_response("", refusal="can't help")])).analyze(_req())
    assert res.refused and "can't help" in res.error
    res = OpenAIVLM(client=FakeOpenAI([HTTPError("rate limited", 429)])).analyze(_req())
    assert res.error and "rate limited" in res.error


def test_openai_list_models_filters_non_vision():
    assert OpenAIVLM(client=FakeOpenAI([])).list_models() == ["gpt-6.1-sol", "gpt-6-luna"]


# ----------------------------------------------------------------- OpenRouter


def _chat(text: str, cost: float | None = None, refusal: str | None = None):
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=50, prompt_tokens_details=None)
    usage.model_extra = {"cost": cost} if cost is not None else {}
    msg = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=usage,
        model="google/gemini-3.8-flash",
    )


class FakeRouter:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

        def create(**kw):
            self.calls.append(kw)
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        img = SimpleNamespace(
            id="google/gemini-3.8-flash",
            model_extra={"architecture": {"input_modalities": ["text", "image", "video"]}},
        )
        txt = SimpleNamespace(
            id="meta/llama-text", model_extra={"architecture": {"input_modalities": ["text"]}}
        )
        self.models = SimpleNamespace(list=lambda: [txt, img])


def test_openrouter_strict_schema_first():
    client = FakeRouter([_chat(json.dumps(VERDICT), cost=0.0012)])
    res = OpenRouterVLM(client=client).analyze(_req())
    assert res.parsed.verdict == "missed" and res.raw["mode"] == "json_schema"
    kw = client.calls[0]
    assert kw["response_format"]["type"] == "json_schema" and kw["response_format"]["json_schema"]["strict"]
    assert "$defs" not in json.dumps(kw["response_format"])
    assert kw["extra_body"]["provider"] == {"require_parameters": True}
    content = kw["messages"][-1]["content"]
    assert content[0]["type"] == "text" and "Was the shot made?" in content[0]["text"]  # text before images
    assert content[1] == {"type": "text", "text": "Sheet R1: tiles R1#1..R1#8"}
    assert content[2]["type"] == "image_url"
    assert res.cost_usd == pytest.approx(0.0012)  # OpenRouter's own figure wins


def test_openrouter_ladder_falls_back_to_json_object_then_repairs():
    client = FakeRouter(
        [HTTPError("No endpoints found", 404), _chat("Sure! {bad json"), _chat(json.dumps(VERDICT))]
    )
    res = OpenRouterVLM(client=client).analyze(_req())
    assert res.parsed is not None and res.raw["mode"] == "json_object+repair"
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert client.calls[1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in client.calls[1]["messages"][-1]["content"][0]["text"]
    assert client.calls[2]["messages"][-2] == {"role": "assistant", "content": "Sure! {bad json"}
    assert res.usage.input_tokens == 1800  # two billed calls


def test_openrouter_fatal_errors_stop_the_ladder():
    res = OpenRouterVLM(client=FakeRouter([HTTPError("bad key", 401)])).analyze(_req())
    assert res.error and "bad key" in res.error


@requires_ffmpeg
def test_openrouter_video_only_for_gemini(testsrc_video):
    client = FakeRouter([_chat("A layup.")])
    req = _req(images=[], output_model=None, video=VideoPart(path=testsrc_video, t0_s=0.5, t1_s=1.5))
    res = OpenRouterVLM(client=client).analyze(req)
    assert res.error is None
    part = client.calls[0]["messages"][-1]["content"][-1]
    assert part["type"] == "video_url" and part["video_url"]["url"].startswith("data:video/mp4;base64,")
    res = OpenRouterVLM(model="openai/gpt-6.1-sol", client=FakeRouter([])).analyze(req)
    assert res.error and "can't take video" in res.error
    assert OpenRouterVLM(client=FakeRouter([])).caps.native_video


def test_openrouter_list_models_keeps_vision_models():
    assert OpenRouterVLM(client=FakeRouter([])).list_models() == ["google/gemini-3.8-flash"]


# ----------------------------------------------------------------- Ollama


class FakeOllama:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def list(self):
        return SimpleNamespace(
            models=[SimpleNamespace(model="gemma4:latest"), SimpleNamespace(model="qwen3.5:4b")]
        )


def _ollama_reply(text: str):
    return SimpleNamespace(
        message=SimpleNamespace(content=text),
        prompt_eval_count=500,
        eval_count=40,
        model="qwen3.5:4b",
        done_reason="stop",
    )


def test_ollama_native_schema_think_off_and_image_cap():
    client = FakeOllama([_ollama_reply(json.dumps(VERDICT))])
    imgs = [ImagePart(jpeg=_jpeg(2000, 1000), label=f"Sheet S{i}") for i in range(12)]
    res = OllamaVLM(client=client).analyze(_req(images=imgs))
    assert res.parsed.verdict == "missed" and res.cost_usd == 0.0 and res.usage.input_tokens == 500
    kw = client.calls[0]
    assert kw["think"] is False and kw["options"]["temperature"] == 0
    assert kw["format"]["properties"]["verdict"]["enum"][0] == "made"
    user = kw["messages"][-1]
    assert len(user["images"]) == 8 and "Image 8: Sheet S11" in user["content"]
    assert max(Image.open(io.BytesIO(user["images"][0])).size) <= 1024


def test_ollama_think_retry_and_unreachable():
    client = FakeOllama([RuntimeError('"gemma" does not support thinking'), _ollama_reply("orange")])
    res = OllamaVLM(client=client).analyze(_req(output_model=None))
    assert res.text == "orange" and "think" not in client.calls[1]
    res = OllamaVLM(client=FakeOllama([ConnectionError("refused")])).analyze(_req())
    assert res.error and "Ollama isn't reachable" in res.error


def test_ollama_list_models():
    assert OllamaVLM(client=FakeOllama([])).list_models() == ["qwen3.5:4b", "gemma4:latest"]


def test_profiles_registered():
    for key, keys in (
        ("openai", ("openai_api_key",)),
        ("openrouter", ("openrouter_api_key",)),
        ("ollama", ()),
    ):
        assert registry.vlms.entry(key).requires_keys == keys


# ----------------------------------------------------------------- live


class Shape(VM):
    color: str
    shape: str


@live
def test_live_openai_parse_and_text():
    vlm = OpenAIVLM()
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
    print("openai parse:", res.model_id, res.usage, res.cost_usd, res.latency_ms, res.parsed, res.stop_reason)
    assert res.error is None, res.error
    assert isinstance(res.parsed, Shape) and "orange" in res.parsed.color.lower()
    res2 = vlm.analyze(
        AnalyzeRequest(
            prompt="In one word, what color is the circle?", images=img, effort="low", max_tokens=2000
        )
    )
    print("openai text:", res2.model_id, res2.usage, res2.cost_usd, repr(res2.text))
    assert res2.error is None and "orange" in res2.text.lower()


@live
def test_live_ollama_if_running():
    vlm = OllamaVLM()
    models = vlm.list_models()
    if models == [vlm.model] and not _ollama_up(vlm):
        pytest.skip("Ollama is not running")
    model = vlm.model if vlm.model in models else models[0]
    res = vlm.analyze(
        AnalyzeRequest(
            prompt="Name the color and shape of the object in Frame A.",
            images=[ImagePart(jpeg=_jpeg(), label="Frame A")],
            output_model=Shape,
            model=model,
            max_tokens=200,
        )
    )
    print("ollama:", res.model_id, res.usage, res.latency_ms, res.parsed, res.error)
    assert res.error is None, res.error
    assert isinstance(res.parsed, Shape)


def _ollama_up(vlm: OllamaVLM) -> bool:
    try:
        vlm.client.list()
        return True
    except Exception:  # noqa: BLE001
        return False
