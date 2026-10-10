from __future__ import annotations

import pytest

from hoops.decide.base import Answer, DecisionResult
from hoops.reason.qa import ROUTE_QUESTION, answer, keyword_route, route_question
from hoops.schemas import Narrative, Provenance
from tests.helpers import requires_ffmpeg
from tests.reason_fakes import FakeVLM, make_run


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("How many shots went in?", "stats"),
        ("What's White's field goal percentage?", "stats"),
        ("Who is winning?", "stats"),
        ("When did White score?", "find_moments"),
        ("Show me every missed three", "find_moments"),
        ("What happened at 0:12?", "explain_play"),
        ("Why did the last shot miss?", "explain_play"),
        ("Give me a recap", "summary"),
        ("Who is #23? What's his name?", "unsupported"),
    ],
)
def test_keyword_routes(question, route):
    assert keyword_route(question) == route


def test_stats_answered_in_code_and_free():
    out = answer(make_run(), "How many shots went in?", vlm=None)
    assert out.route == "stats" and out.cost_usd == 0.0 and out.provider is None
    assert "3 shot attempts, 2 made, 1 missed" in out.answer  # rejected e4 is ignored
    assert "**White**: 5 pts, FG 2/2 (100%), 3PT 1/1, 1 reb" in out.answer
    assert "1 of these shots are not confirmed yet" in out.answer
    assert [c["event_id"] for c in out.citations] == ["e1", "e2", "e5"]
    assert out.citations[0] == {"t_s": 12.4, "event_id": "e1", "label": "Made jumper (White), 0:12"}


def test_find_moments_filters_by_result_kind_and_team():
    out = answer(make_run(), "When did White score?", vlm=None)
    assert out.route == "find_moments" and [c["event_id"] for c in out.citations] == ["e1", "e5"]
    assert "- 0:50: Made 3-pointer (White) (provisional)" in out.answer
    out = answer(make_run(), "Show me every missed three", vlm=None)
    assert [c["event_id"] for c in out.citations] == ["e2"]
    assert out.citations[0]["label"] == "Missed 3-pointer (Blue), 0:30"
    out = answer(make_run(), "When were the dunks?", vlm=None)
    assert out.citations == [] and "couldn't find" in out.answer and "Rebound" in out.answer


def test_unsupported_and_empty():
    out = answer(make_run(), "What's the real name of #23?", vlm=FakeVLM())
    assert out.route == "unsupported" and "never identify people" in out.answer
    assert answer(make_run(), "  ", vlm=None).route == "unsupported"


def test_explain_without_vlm_uses_event_log():
    out = answer(make_run(), "What happened at 0:12?", vlm=None)
    assert out.route == "explain_play" and out.citations[0]["event_id"] == "e1"
    assert "event log says: Made jumper (White), 0:12 (confirmed, confidence 90%)" in out.answer


@requires_ffmpeg
def test_explain_play_sends_window_sheets(testsrc_video):
    vlm = FakeVLM(text="The ball rolls off the rim in W1#2.")
    out = answer(
        make_run(), "What happened at 0:01?", vlm=vlm, video=testsrc_video, history=[("Hi", "Hello")]
    )
    assert out.route == "explain_play" and out.provider == "fake" and out.cost_usd == pytest.approx(0.01)
    req = vlm.requests[0]
    assert req.purpose == "qa" and req.images and req.images[0].label.startswith("Sheet W1")
    assert "What happened at 0:01?" in req.prompt and req.history == [("Hi", "Hello")]
    assert req.cache_key.startswith("run1:qa:") and '"events":' in req.cv_context
    assert any(c["label"].startswith("Frame W1#2") for c in out.citations)


class FakeDecider:
    key = "jev"
    supports_images = False

    def __init__(self, value="find_moments", error=None):
        self.value, self.error, self.calls = value, error, []

    def decide(self, state, questions, images=()):
        self.calls.append((state, questions))
        if self.error:
            return DecisionResult(answers={}, decider="jev", error=self.error)
        ans = Answer(kind="choice", value=self.value, probabilities={self.value: 0.93}, confidence=0.93)
        return DecisionResult(
            answers={"route": ans}, decider="jev", model="jev-latest", latency_ms=80, cost_usd=1e-5
        )


def test_decider_routes_and_records_decision():
    decider = FakeDecider("find_moments")
    out = answer(make_run(), "Anything from White that went in?", vlm=None, decider=decider)
    assert out.route == "find_moments" and out.decision["decider"] == "jev"
    assert out.decision["route_confidence"] == 0.93 and out.cost_usd == pytest.approx(1e-5)
    state, questions = decider.calls[0]
    assert state == {"question": "Anything from White that went in?"} and questions == {
        "route": ROUTE_QUESTION
    }


def test_decider_failure_falls_back_to_keywords():
    route, decision = route_question("How many shots?", FakeDecider(error="timeout"))
    assert route == "stats" and decision["fallback"] == "keywords" and decision["error"] == "timeout"
    route, decision = route_question("How many shots?", FakeDecider(value="something_else"))
    assert route == "stats" and decision["fallback"] == "keywords"


def test_summary_reuses_saved_narrative_or_falls_back():
    run = make_run()
    out = answer(run, "Give me a recap", vlm=None)
    assert out.route == "summary" and "No AI summary is saved" in out.answer
    run.narratives.append(
        Narrative(
            kind="recap",
            text="White pulled away late.",
            cited_event_ids=["e5"],
            provenance=Provenance(source="vlm", producer="anthropic:claude-opus-5-5", stage="summarize"),
        )
    )
    out = answer(run, "Give me a recap", vlm=FakeVLM())
    assert out.answer == "White pulled away late." and out.cost_usd == 0.0
    assert (out.provider, out.model) == ("anthropic", "claude-opus-5-5")
    assert out.citations[0]["event_id"] == "e5"
