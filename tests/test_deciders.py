from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hoops import registry
from hoops.decide.adapter import FALLBACK_BETA, AdapterDecider, claude_provider
from hoops.decide.base import Choice, Decider, Noul, Score
from hoops.decide.jev import JevDecider
from hoops.decide.openai_decisions import OpenAIDecisionsDecider, data_url
from hoops.decide.questions import EVENT_TYPING, QA_ROUTE, TRIAGE, event_state, qa_state, triage_state
from hoops.decide.rules import RulesDecider
from hoops.events.shots import detect_candidates
from tests.helpers import live
from tests.shot_fixtures import arc, detections, held

QS = {
    "ok": Noul("The ball went in."),
    "kind": Choice("Which shot?", {"jumper": "from range", "layup": None}),
    "level": Score("How clear is it?", ["unclear", "somewhat", "very clear"]),
}


@pytest.fixture(scope="module")
def make_cand():
    return detect_candidates(detections(held(0.0, 1.0) + arc(1.0)))[0]


# ------------------------------------------------------------------ rules


def test_rules_event_typing(make_cand):
    d = RulesDecider()
    assert isinstance(d, Decider)
    r = d.decide(event_state(make_cand, with_scores=True), EVENT_TYPING)
    assert r.error is None and r.cost_usd == 0.0
    assert r.answers["is_shot_attempt"].value == pytest.approx(make_cand.attempt_score, abs=1e-3)
    out = r.answers["outcome"]
    assert out.value == "made" and out.confidence == pytest.approx(make_cand.make_score, abs=1e-3)
    assert set(out.probabilities) == {"made", "missed", "unclear"}
    # Without numeric scores it falls back to the descriptors.
    r2 = d.decide(event_state(make_cand), EVENT_TYPING)
    assert r2.answers["outcome"].value == "made" and r2.answers["is_shot_attempt"].value >= 0.8


def test_rules_unknown_question_is_uninformative():
    r = RulesDecider().decide({"x": "y"}, QS)
    assert r.answers["ok"].value == 0.5 and r.answers["ok"].confidence is None
    assert r.answers["kind"].confidence == 0.0
    assert r.answers["kind"].probabilities == {"jumper": 0.5, "layup": 0.5}
    assert r.answers["level"].value == "somewhat" and r.answers["level"].confidence == 0.0


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("How many shots did team A make?", "stats"),
        ("Show me every made three", "find_moments"),
        ("Why did that layup miss?", "explain_play"),
        ("Give me a quick recap", "summary"),
        ("What's the weather?", "unsupported"),
    ],
)
def test_rules_qa_route(question, route):
    r = RulesDecider().decide(qa_state(question, "events: 4 shots"), QA_ROUTE)
    assert r.answers["route"].value == route


def test_rules_triage():
    r = RulesDecider().decide(
        triage_state(camera_motion="steady", scene_cuts="none", rim="one, steady"), TRIAGE
    )
    assert r.answers["footage_type"].value == "fixed_camera"
    r = RulesDecider().decide(
        triage_state(scoreboard="visible", scene_cuts="many", court_lines="visible"), TRIAGE
    )
    assert r.answers["footage_type"].value == "broadcast"
    assert r.answers["court_lines_visible"].value == 0.8


def test_states_are_small_and_categorical(make_cand):
    s = event_state(make_cand)
    assert "rule_scores" not in s and all(isinstance(v, str) for v in s.values())
    assert len(json.dumps(s)) < 1000
    assert triage_state(a=None, b="", c="x" * 1000)["c"].endswith("…")


# ------------------------------------------------------------------ jev (wire-level, no key needed)


def _jev_with_transport(handler):
    import httpx2
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0)
    )
    return JevDecider(client=client)


def test_jev_request_and_answer_mapping():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = {
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 1000, "output_tokens": 9},
            "answers": {
                "ok": {"type": "noul", "noul": 0.91},
                "kind": {
                    "type": "choice",
                    "choice": "layup",
                    "confidence": 0.7,
                    "probabilities": {"jumper": 0.2, "layup": 0.8},
                },
                "level": {
                    "type": "score",
                    "score": 1.7,
                    "confidence": 0.8,
                    "legend": {"0": "unclear", "1": "somewhat", "2": "very clear"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                },
            },
        }
        return __import__("httpx2").Response(200, json=body)

    r = _jev_with_transport(handler).decide({"crossing": "inside rim edges"}, QS)
    assert r.error is None
    assert seen["url"].endswith("/v1/systemone")
    body = seen["body"]
    assert body["model"] == "jev-latest" and body["state"] == {"crossing": "inside rim edges"}
    assert body["questions"]["ok"] == {"type": "noul", "instructions": "The ball went in."}
    assert body["questions"]["kind"] == {
        "type": "choice",
        "instructions": "Which shot?",
        "criteria": {"jumper": "from range", "layup": None},
    }
    assert body["questions"]["level"]["criteria"] == ["unclear", "somewhat", "very clear"]
    assert (
        r.answers["ok"].kind == "noul"
        and r.answers["ok"].value == 0.91
        and r.answers["ok"].confidence is None
    )
    assert r.answers["kind"].value == "layup" and r.answers["kind"].confidence == 0.7
    lvl = r.answers["level"]
    assert (
        lvl.value == "very clear"
        and lvl.probabilities["very clear"] == 0.8
        and lvl.raw == {"expected_score": 1.7}
    )
    assert r.model == "jev-1.13.0"
    assert r.cost_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_jev_errors_become_results():
    import httpx2

    r = _jev_with_transport(lambda req: httpx2.Response(401, json={"detail": "bad key"})).decide("s", QS)
    assert r.answers == {}
    assert r.error and ("401" in r.error or "Authentication" in r.error)
    assert "test-key" not in r.error


def test_jev_factory_requires_key(monkeypatch):
    from hoops import config

    monkeypatch.setenv("TYPESAFE_API_KEY", "")  # env beats .env; empty means "no key"
    config.get_settings.cache_clear()
    try:
        assert "TYPESAFE_API_KEY" in (registry.deciders.entry("jev").unavailable_reason() or "")
    finally:
        config.get_settings.cache_clear()


# ------------------------------------------------------------------ adapter


class FakeProvider:
    model_name = "fake-llm"

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def request(self, messages, *, schema, structured):
        from system_one_adapter.providers import ProviderResult

        self.calls.append({"messages": messages, "schema": schema, "structured": structured})
        return ProviderResult(text=json.dumps(self.payload), input_tokens=100, output_tokens=20)

    def translate_error(self, error):
        from typesafe_sdk import TypeSafeError

        return TypeSafeError(str(error))


def test_adapter_maps_probabilities():
    payload = {
        "answers": {"ok": 0.8, "kind": {"jumper": 0.3, "layup": 0.7}, "level": {"0": 0.1, "1": 0.2, "2": 0.7}}
    }
    fake = FakeProvider(payload)
    d = AdapterDecider(
        provider="anthropic", model="claude-opus-5-5", key="adapter:claude", provider_instance=fake
    )
    r = d.decide({"crossing": "inside rim edges"}, QS)
    assert r.error is None, r.error
    assert fake.calls and fake.calls[0]["structured"] is True
    assert "inside rim edges" in fake.calls[0]["messages"][1].content
    assert r.answers["ok"].value == pytest.approx(0.8)
    assert r.answers["kind"].value == "layup" and r.answers["kind"].probabilities["layup"] == pytest.approx(
        0.7
    )
    assert r.answers["kind"].confidence == pytest.approx(0.4)  # (0.7 - 1/2) / (1 - 1/2)
    assert r.answers["level"].value == "very clear"
    assert r.cost_usd == pytest.approx((100 * 4.0 + 20 * 20.0) / 1e6)


def test_adapter_error_is_reported():
    fake = FakeProvider({"answers": {"ok": "not a number"}})
    d = AdapterDecider(provider="openai", model="gpt-6.1-sol", key="adapter:openai", provider_instance=fake)
    r = d.decide("state", {"ok": QS["ok"]})
    assert r.answers == {} and r.error


def test_claude_provider_request_kwargs():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        text = json.dumps({"answers": {"ok": 0.9}})
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=50, output_tokens=200),
            model_dump=lambda **_: {"stub": True},
        )

    p = claude_provider("claude-opus-5-5", api_key="test-key")
    p._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    d = AdapterDecider(
        provider="anthropic", model="claude-opus-5-5", key="adapter:claude", provider_instance=p
    )
    r = d.decide({"crossing": "inside rim edges"}, {"ok": QS["ok"]})
    assert r.error is None, r.error
    assert r.answers["ok"].value == pytest.approx(0.9)
    assert captured["model"] == "claude-opus-5-5" and captured["max_tokens"] == 16000
    assert captured["output_config"]["effort"] == "low"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["betas"] == [FALLBACK_BETA] and captured["fallbacks"] == "default"
    assert not {"temperature", "thinking", "top_p"} & set(captured)
    assert captured["messages"][-1]["role"] == "user"  # no assistant prefill


# ------------------------------------------------------------------ openai decisions


def _decision(answers):
    from openai.types.decision import Decision

    return Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": answers,
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 10,
                "total_tokens": 2010,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def test_openai_decisions_mapping_and_images():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _decision(
            [
                {"type": "predicate", "name": "ok", "probability": 0.93},
                {
                    "type": "choice",
                    "name": "kind",
                    "choice": "jumper",
                    "confidence": 0.6,
                    "probabilities": [
                        {"value": "jumper", "probability": 0.8},
                        {"value": "layup", "probability": 0.2},
                    ],
                },
                {"type": "refusal", "name": "level"},
            ]
        )

    jpeg = b"\xff\xd8\xff\xe0fakejpeg"
    d = OpenAIDecisionsDecider(client=SimpleNamespace(decisions=SimpleNamespace(create=create)))
    r = d.decide({"crossing": "inside rim edges"}, QS, images=[jpeg])
    assert r.error is None
    qs = captured["questions"]
    assert qs[0] == {"type": "predicate", "name": "ok", "instructions": "The ball went in."}
    assert qs[1]["choices"] == [{"value": "jumper", "description": "from range"}, {"value": "layup"}]
    assert qs[2]["levels"] == [{"label": "unclear"}, {"label": "somewhat"}, {"label": "very clear"}]
    msg = captured["input"][0]
    assert msg["role"] == "user" and msg["content"][0]["type"] == "input_text"
    assert msg["content"][1]["image_url"] == data_url(jpeg) and data_url(jpeg).startswith(
        "data:image/jpeg;base64,"
    )
    assert r.answers["ok"].value == 0.93
    assert r.answers["kind"].value == "jumper" and r.answers["kind"].probabilities == {
        "jumper": 0.8,
        "layup": 0.2,
    }
    assert "level" not in r.answers  # refusal
    assert r.cost_usd == pytest.approx((2000 * 0.10 + 10 * 0.50) / 1e6)
    # Text-only input is a plain string.
    d.decide("plain state", {"ok": QS["ok"]})
    assert captured["input"] == "plain state"


def test_registry_entries():
    reg = registry.deciders
    for key in ("rules", "jev", "adapter:claude", "adapter:openai", "adapter:gemini", "openai-decisions"):
        e = reg.entry(key)
        assert "supports_images" in e.extra and e.label
    assert reg.entry("openai-decisions").extra["supports_images"] is True
    assert reg.entry("jev").requires_modules == ("typesafe_sdk",)
    assert reg.entry("adapter:claude").requires_keys == ("anthropic_api_key",)
    assert isinstance(reg.create("rules"), RulesDecider)


# ------------------------------------------------------------------ live (HOOPS_LIVE=1; tiny, no footage)

LIVE_STATE = {"ball_path": "rose above rim, then dropped through rim region", "crossing": "inside rim edges"}
LIVE_Q = {"made": Noul("The shot went in.")}


@live
@pytest.mark.parametrize("key", ["adapter:claude", "adapter:openai", "openai-decisions"])
def test_live_tiny_call(key):
    r = registry.deciders.create(key).decide(LIVE_STATE, LIVE_Q)
    assert r.error is None, r.error
    a = r.answers["made"]
    assert a.kind == "noul" and 0.0 <= float(a.value) <= 1.0
    print(f"\n{key}: model={r.model} p={a.value:.3f} latency_ms={r.latency_ms:.0f} cost_usd={r.cost_usd}")
