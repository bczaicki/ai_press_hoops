from __future__ import annotations

from dataclasses import replace

import pytest

from hoops.decide.base import PRESETS, Answer, DecisionResult
from hoops.decide.questions import triage_state
from hoops.decide.rules import RulesDecider
from hoops.events.escalate import apply_verdicts, audit_pick, recompute, triage, type_candidates
from hoops.events.shots import Candidate, detect_candidates
from hoops.events.stats import compute_stats
from hoops.schemas import DecisionRecord, EventType, TeamRef, WindowVerdict
from tests.shot_fixtures import arc, detections, held

BAL = PRESETS["balanced"]


def _cand(i: int, t: float = 0.0) -> Candidate:
    t = t or 2.0 + 3 * i
    return Candidate(
        cand_id=f"cand_{round(t * 1000):07d}",
        t_start_s=t - 2,
        t_key_s=t,
        t_end_s=t + 1.5,
        attempt_score=0.9,
        make_score=0.9,
        features={"t_apex_s": t - 0.4},
        descriptors={"crossing": "inside rim edges"},
        rim_box=(620, 300, 660, 316),
        shooter_track_id=7,
        shooter_xy=(400.0, 560.0),
        ball_points=[],
        near_scene_cut=False,
        shooter_team_id="A",
    )


class FakeDecider:
    """Answers EVENT_TYPING from a list of (p_attempt, outcome, outcome_confidence), in call order."""

    def __init__(self, script, key="jev", supports_images=False):
        self.script, self.key, self.supports_images, self.calls = list(script), key, supports_images, []

    def decide(self, state, questions, images=()):
        self.calls.append({"state": state, "images": list(images)})
        item = self.script.pop(0)
        if isinstance(item, str):
            return DecisionResult({}, self.key, "fake-1", error=item)
        p, outcome, conf = item
        rest = (1 - conf) / 2
        probs = {o: (conf if o == outcome else rest) for o in ("made", "missed", "unclear")}
        return DecisionResult(
            {
                "is_shot_attempt": Answer("noul", p),
                "outcome": Answer("choice", outcome, probabilities=probs, confidence=conf),
                "shot_kind": Answer("choice", "jumper", probabilities={"jumper": 0.8}, confidence=0.8),
            },
            self.key,
            "fake-1",
            latency_ms=5.0,
            cost_usd=1e-6,
        )


def test_accept_reject_escalate():
    cands = [_cand(0), _cand(1), _cand(2), _cand(3)]
    d = FakeDecider([(0.95, "made", 0.95), (0.05, "missed", 0.9), (0.5, "made", 0.9), (0.95, "made", 0.6)])
    events, records, esc = type_candidates(cands, d, preset=BAL, run_id="r1", audit_rate=0.0)
    assert [e.status for e in events] == ["confirmed", "rejected", "needs_review", "needs_review"]
    assert [e.type for e in events] == [
        EventType.made_shot,
        EventType.missed_shot,
        EventType.made_shot,
        EventType.made_shot,
    ]
    assert [c.cand_id for c in esc] == [cands[2].cand_id, cands[3].cand_id]
    e0 = events[0]
    assert e0.confidence_kind == "calibrated" and e0.provenance.source == "system_one"
    assert e0.provenance.producer == "jev:fake-1" and e0.shot.result == "made" and e0.shot.kind == "jumper"
    assert e0.shot.image_x == 400.0 and e0.team_id == "A" and e0.player_ids == ["trk_7"]
    assert e0.confidence == pytest.approx(0.95 * 0.95, abs=1e-3)
    assert e0.event_id == "shot_0002000" and e0.features["cand_id"] == "cand_0002000"
    assert e0.features["system_one"]["p_attempt"] == 0.95
    assert len(records) == 4 and all(isinstance(r, DecisionRecord) for r in records)
    assert [r.routed for r in records] == ["accept", "reject", "escalate", "escalate"]
    assert records[0].questions["outcome"]["kind"] == "choice"
    assert records[0].answers["is_shot_attempt"] == {
        "kind": "noul",
        "value": 0.95,
        "probabilities": None,
        "confidence": None,
    }
    assert "rule_scores" not in records[0].state  # hosted deciders never see numbers


def test_unclear_outcome_is_a_shot_attempt_needing_review():
    events, _, esc = type_candidates(
        [_cand(0)], FakeDecider([(0.97, "unclear", 0.9)]), preset=BAL, run_id="r"
    )
    assert events[0].type == EventType.shot_attempt and events[0].shot.result == "unknown"
    assert events[0].status == "needs_review" and len(esc) == 1


def test_audit_sampling_is_deterministic():
    ids = [f"cand_{i:07d}" for i in range(2000)]
    picks = [audit_pick(i, 0.1) for i in ids]
    assert picks == [audit_pick(i, 0.1) for i in ids]
    assert 0.07 < sum(picks) / len(picks) < 0.13
    assert not any(audit_pick(i, 0.0) for i in ids[:50]) and all(audit_pick(i, 1.0) for i in ids[:50])
    cands = [_cand(i) for i in range(3)]
    events, _, esc = type_candidates(
        cands, FakeDecider([(0.99, "made", 0.99)] * 3), preset=BAL, run_id="r", audit_rate=1.0
    )
    assert len(esc) == 3 and all(e.status == "confirmed" and e.features["audit"] for e in events)


def test_escalation_cap_prefers_most_uncertain():
    cands = [_cand(i) for i in range(5)]
    script = [
        (0.6, "made", 0.9),
        (0.5, "made", 0.9),
        (0.8, "made", 0.9),
        (0.45, "made", 0.9),
        (0.99, "made", 0.99),
    ]
    events, _, esc = type_candidates(
        cands, FakeDecider(script), preset=BAL, run_id="r", max_escalations=2, audit_rate=1.0
    )
    assert [c.cand_id for c in esc] == [
        cands[1].cand_id,
        cands[3].cand_id,
    ]  # p=0.5 and p=0.45; no room for audits
    assert events[0].features.get("escalation") == "capped" and events[0].status == "needs_review"
    assert not events[4].features["audit"] and events[4].status == "confirmed"


def test_images_only_for_image_deciders():
    seen = []

    def images_for(c):
        seen.append(c.cand_id)
        return [b"jpeg"]

    text_only = FakeDecider([(0.99, "made", 0.99)])
    type_candidates([_cand(0)], text_only, preset=BAL, run_id="r", images_for=images_for)
    assert text_only.calls[0]["images"] == [] and seen == []
    vision = FakeDecider([(0.99, "made", 0.99)], key="openai-decisions", supports_images=True)
    type_candidates([_cand(0)], vision, preset=BAL, run_id="r", images_for=images_for)
    assert vision.calls[0]["images"] == [b"jpeg"] and seen == ["cand_0002000"]


def test_decider_error_falls_back_to_rules():
    events, records, _ = type_candidates([_cand(0)], FakeDecider(["HTTP 503"]), preset=BAL, run_id="r")
    assert [r.decider for r in records] == ["jev:fake-1", "rules:rules-v1"]
    assert records[0].error == "HTTP 503" and records[0].routed is None and records[1].routed is not None
    e = events[0]
    assert e.provenance.source == "rule" and e.confidence_kind == "heuristic"
    assert e.features["system_one"]["fallback"] == "rules"


def test_recompute_moves_statuses_with_presets():
    cands = [_cand(0), _cand(1)]
    events, _, _ = type_candidates(
        cands, FakeDecider([(0.9, "made", 0.9), (0.2, "missed", 0.9)]), preset=BAL, run_id="r"
    )
    assert [e.status for e in events] == ["confirmed", "needs_review"]
    assert [e.status for e in recompute(events, PRESETS["strict"])] == ["needs_review", "rejected"]
    assert [e.status for e in recompute(events, PRESETS["eager"])] == ["confirmed", "needs_review"]
    assert recompute(events, PRESETS["strict"])[0].features["preset"] == "strict"
    assert events[0].status == "confirmed"  # inputs are not mutated
    reviewed = events[0].model_copy(update={"review": "accepted"})
    assert recompute([reviewed], PRESETS["strict"])[0].status == "confirmed"


def _verdict(v, kind="unknown", conf=0.8):
    return WindowVerdict(
        verdict=v,
        shot_kind=kind,
        confidence=conf,
        evidence_tiles=["S1#4"],
        shooter_description=None,
        reason="ball in net",
    )


def test_apply_verdicts_confirm_flip_reject():
    cands = [_cand(i) for i in range(4)]
    script = [(0.5, "made", 0.9), (0.95, "made", 0.6), (0.5, "missed", 0.9), (0.97, "unclear", 0.9)]
    events, _, _ = type_candidates(cands, FakeDecider(script), preset=BAL, run_id="r")
    verdicts = {
        events[0].event_id: _verdict("made", "layup"),
        cands[1].cand_id: _verdict("missed", conf=1.7),  # keyed by cand_id; confidence clamped
        events[2].event_id: _verdict("no_shot"),
        events[3].event_id: _verdict("unclear"),
    }
    out = apply_verdicts(events, verdicts, producer="anthropic:claude-opus-5-5")
    assert [e.status for e in out] == ["confirmed", "confirmed", "rejected", "needs_review"]
    assert out[0].type == EventType.made_shot and out[0].shot.kind == "layup"
    assert (
        out[1].type == EventType.missed_shot and out[1].shot.result == "missed" and out[1].confidence == 1.0
    )
    assert out[0].provenance.source == "vlm" and out[0].provenance.stage == "adjudicate"
    assert out[0].features["vlm"]["before"]["status"] == "needs_review"
    assert out[0].features["system_one"]["p_attempt"] == 0.5  # original scores kept
    assert out[1].features["vlm"]["agrees_with_system_one"] is False
    assert out[0].evidence[-1].tile_id == "S1#4"
    # VLM-ruled events keep their status when re-gated.
    assert [e.status for e in recompute(out, PRESETS["eager"])] == [e.status for e in out]


def test_triage():
    footage, rec = triage(RulesDecider(), triage_state(camera_motion="steady", rim="one, steady"), run_id="r")
    assert (
        footage == "fixed_camera"
        and rec.purpose == "triage"
        and rec.answers["footage_type"]["value"] == "fixed_camera"
    )
    footage, rec = triage(FakeDecider(["down"]), {"x": "y"})
    assert footage == "unknown" and rec.error == "down"


def test_end_to_end_rules_pipeline():
    cands = detect_candidates(detections(held(0.0, 1.0) + arc(1.0)))
    events, records, esc = type_candidates(cands, RulesDecider(), preset=BAL, run_id="r", audit_rate=0.0)
    assert len(events) == 1 and events[0].status == "confirmed" and events[0].type == EventType.made_shot
    assert events[0].provenance.source == "rule" and records[0].state["rule_scores"]["make"] > 0.9
    assert esc == []
    [team_a, _unknown] = compute_stats(events, [TeamRef(team_id="A")])
    assert (team_a.stats.fga, team_a.stats.fgm, team_a.stats.pts) == (1, 1, 2)
    # Strict re-gating sends the 0.90 attempt score to review without re-running anything.
    assert recompute(events, replace(PRESETS["strict"]))[0].status == "needs_review"
