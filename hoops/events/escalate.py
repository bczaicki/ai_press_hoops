"""System One routing for shot candidates: type, gate, audit, escalate, and re-gate.

For each candidate:

1. Ask EVENT_TYPING, attaching images when the decider supports them and ``images_for`` is given.
2. Gate with ``hoops.decide.base.gate``. ``is_shot_attempt`` (Noul p) can reject, escalate, or
   accept. An accepted attempt is only accepted overall when ``outcome`` is made or missed with
   confidence at or above the accept bound. Otherwise the outcome goes to System Two.
3. Accepted calls are audited at ``audit_rate``: a deterministic hash of the cand_id picks them,
   and they are sent to System Two too while staying confirmed.
4. Escalations are capped at ``max_escalations``, most uncertain first and audits last. The rest
   stay ``needs_review`` for the human Review tab.

Status mapping: accept -> confirmed, reject -> rejected, escalate -> needs_review. If a hosted
decider fails, the rules baseline answers instead. The failed call is still recorded, and the
event's provenance says "rule".
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, get_args

from hoops.decide.base import Answer, Decider, DecisionResult, Preset, Route, gate, question_spec
from hoops.decide.common import error_text
from hoops.decide.questions import EVENT_TYPING, TRIAGE, event_state
from hoops.decide.rules import RulesDecider
from hoops.events.shots import Candidate
from hoops.schemas import (
    DecisionRecord,
    Event,
    EventType,
    Evidence,
    FootageType,
    Provenance,
    ShotInfo,
    ShotKind,
    WindowVerdict,
)

CALIBRATED = frozenset({"jev", "openai-decisions"})
STATUS: dict[str, str] = {"accept": "confirmed", "reject": "rejected", "escalate": "needs_review"}
_SHOT_KINDS = set(get_args(ShotKind))
_FOOTAGE = set(get_args(FootageType))


def _safe_decide(
    decider: Decider, state: Any, questions: dict, images: Sequence[bytes] = ()
) -> DecisionResult:
    try:
        return decider.decide(state, questions, images)
    except Exception as exc:  # noqa: BLE001 - a decider that breaks its contract must not stop the run
        return DecisionResult({}, getattr(decider, "key", "?"), error=error_text(exc))


def _label(res: DecisionResult) -> str:
    return f"{res.decider}:{res.model}" if res.model else res.decider


def _record(
    run_id: str,
    subject: str,
    purpose: str,
    res: DecisionResult,
    state: Any,
    questions: dict,
    routed: Route | None,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=f"{run_id}:{subject}:{purpose}:{res.decider}",
        decider=_label(res),
        purpose=purpose,  # type: ignore[arg-type]
        subject_id=subject,
        state=state,
        questions={k: question_spec(q) for k, q in questions.items()},
        answers={k: a.spec() for k, a in res.answers.items()},
        routed=routed,
        latency_ms=res.latency_ms,
        cost_usd=res.cost_usd,
        error=res.error,
    )


def audit_pick(cand_id: str, rate: float) -> bool:
    """Deterministic audit sampling: the same cand_id is always (or never) audited at a given rate."""
    return int(hashlib.sha256(cand_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def summarize(res: DecisionResult, *, fallback: bool = False) -> dict[str, Any]:
    """The numbers recompute() needs, stored in Event.features['system_one']."""
    a, o, k = (res.answers.get(n) for n in ("is_shot_attempt", "outcome", "shot_kind"))
    return {
        "decider": res.decider,
        "model": res.model,
        "fallback": "rules" if fallback else None,
        "p_attempt": None if a is None else float(a.value),
        "outcome": None if o is None else str(o.value),
        "outcome_probs": None if o is None else o.probabilities,
        "outcome_confidence": None if o is None else o.confidence,
        "shot_kind": None if k is None else str(k.value),
        "shot_kind_probs": None if k is None else k.probabilities,
    }


def route(s1: dict[str, Any], preset: Preset) -> Route:
    """Gate stored System One answers with a preset (no inference)."""
    p = s1.get("p_attempt")
    if p is None:
        return "escalate"
    r = gate(Answer("noul", float(p)), preset.accept, preset.reject)
    if r != "accept":
        return r
    outcome = s1.get("outcome")
    if outcome not in ("made", "missed"):
        return "escalate"
    return gate(
        Answer("choice", outcome, confidence=s1.get("outcome_confidence")), preset.accept, preset.reject
    )


def _uncertainty(s1: dict[str, Any]) -> float:
    p = s1.get("p_attempt")
    u_attempt = 1.0 if p is None else 1 - abs(2 * float(p) - 1)
    u_outcome = 1 - float(s1.get("outcome_confidence") or 0.0)
    return max(u_attempt, u_outcome)


def _shot_fields(s1: dict[str, Any]) -> tuple[EventType, str, float]:
    p = float(s1.get("p_attempt") or 0.0)
    outcome = s1.get("outcome")
    if outcome in ("made", "missed"):
        p_out = (s1.get("outcome_probs") or {}).get(outcome, s1.get("outcome_confidence") or 0.0)
        etype = EventType.made_shot if outcome == "made" else EventType.missed_shot
        return etype, outcome, min(1.0, max(0.0, p * float(p_out)))
    return EventType.shot_attempt, "unknown", min(1.0, max(0.0, p))


def _describe(etype: EventType, kind: str) -> str:
    noun = "shot" if kind == "unknown" else kind.replace("_", " ")
    return {EventType.made_shot: f"Made {noun}", EventType.missed_shot: f"Missed {noun}"}.get(
        etype, f"Shot attempt ({noun}, outcome unclear)"
    )


def build_event(c: Candidate, s1: dict[str, Any], r: Route, preset: Preset) -> Event:
    etype, result, conf = _shot_fields(s1)
    kind = s1.get("shot_kind") if s1.get("shot_kind") in _SHOT_KINDS else "unknown"
    decider = s1.get("decider") or "?"
    evidence = [Evidence(t_s=c.t_key_s, note="ball at the rim plane")]
    if c.features.get("t_apex_s") is not None:
        evidence.append(Evidence(t_s=float(c.features["t_apex_s"]), note="top of the ball's arc"))
    return Event(
        event_id=c.cand_id.replace("cand_", "shot_", 1),
        type=etype,
        t_start_s=c.t_start_s,
        t_end_s=c.t_end_s,
        t_key_s=c.t_key_s,
        team_id=c.shooter_team_id,
        player_ids=[f"trk_{c.shooter_track_id}"] if c.shooter_track_id is not None else [],
        shot=ShotInfo(
            result=result,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            image_x=None if c.shooter_xy is None else c.shooter_xy[0],
            image_y=None if c.shooter_xy is None else c.shooter_xy[1],
        ),
        confidence=round(conf, 4),
        confidence_kind="calibrated" if decider in CALIBRATED else "heuristic",
        status=STATUS[r],  # type: ignore[arg-type]
        description=_describe(etype, kind),
        features={
            "cand_id": c.cand_id,
            "trigger": c.trigger,
            "rule": dict(c.features),
            "descriptors": dict(c.descriptors),
            "system_one": s1,
            "route": r,
            "preset": preset.name,
            "near_scene_cut": c.near_scene_cut,
            "audit": False,
            "escalated": False,
        },
        evidence=evidence,
        provenance=Provenance(
            source="rule" if decider == "rules" else "system_one",
            producer=f"{decider}:{s1.get('model')}",
            stage="decide",
            params={"preset": preset.name, "accept": preset.accept, "reject": preset.reject},
        ),
    )


def type_candidates(
    candidates: Sequence[Candidate],
    decider: Decider,
    *,
    preset: Preset,
    run_id: str,
    audit_rate: float = 0.1,
    max_escalations: int = 20,
    images_for: Callable[[Candidate], list[bytes]] | None = None,
) -> tuple[list[Event], list[DecisionRecord], list[Candidate]]:
    """Type every candidate with System One; returns (events, decision records, candidates to escalate)."""
    events: list[Event] = []
    records: list[DecisionRecord] = []
    queue: list[tuple[tuple[int, float, float], int]] = []  # (priority, index into events)
    use_images = bool(getattr(decider, "supports_images", False) and images_for is not None)
    rules = decider if isinstance(decider, RulesDecider) else None
    for c in candidates:
        state = event_state(c, with_scores=bool(getattr(decider, "wants_scores", False)))
        images = list(images_for(c)) if use_images and images_for else []
        res = _safe_decide(decider, state, EVENT_TYPING, images)
        fell_back = bool(res.error or not res.answers)
        if fell_back:
            records.append(_record(run_id, c.cand_id, "event_typing", res, state, EVENT_TYPING, None))
            rules = rules or RulesDecider()
            state = event_state(c, with_scores=True)
            res = rules.decide(state, EVENT_TYPING)
        s1 = summarize(res, fallback=fell_back and not isinstance(decider, RulesDecider))
        r = route(s1, preset)
        records.append(_record(run_id, c.cand_id, "event_typing", res, state, EVENT_TYPING, r))
        events.append(build_event(c, s1, r, preset))
        if r == "escalate":
            queue.append(((0, -_uncertainty(s1), c.t_key_s), len(events) - 1))
        elif r == "accept" and audit_pick(c.cand_id, audit_rate):
            queue.append(((1, 0.0, c.t_key_s), len(events) - 1))
    queue.sort()
    chosen = {i for _, i in queue[: max(0, max_escalations)]}
    for prio, i in queue:
        f = events[i].features
        f["audit"] = prio[0] == 1 and i in chosen
        f["escalated"] = i in chosen
        if i not in chosen and prio[0] == 0:
            f["escalation"] = "capped"  # stays needs_review for a human
    by_id = {c.cand_id: c for c in candidates}
    escalate = sorted((by_id[events[i].features["cand_id"]] for i in chosen), key=lambda c: c.t_key_s)
    return events, records, escalate


def apply_verdicts(events: list[Event], verdicts: dict[str, WindowVerdict], *, producer: str) -> list[Event]:
    """Apply System Two verdicts (keyed by event_id or cand_id): confirm, flip, or reject events.

    Original System One features are kept; the verdict is added under features['vlm']. Events a
    human already reviewed are left untouched.
    """
    out: list[Event] = []
    for e in events:
        v = verdicts.get(e.event_id) or verdicts.get(str(e.features.get("cand_id")))
        if v is None or e.review != "unreviewed":
            out.append(e)
            continue
        new = e.model_copy(deep=True)
        before = {
            "type": e.type.value,
            "status": e.status,
            "confidence": e.confidence,
            "source": e.provenance.source,
        }
        shot = new.shot or ShotInfo()
        if v.verdict == "made":
            new.type, shot.result, new.status = EventType.made_shot, "made", "confirmed"
        elif v.verdict == "missed":
            new.type, shot.result, new.status = EventType.missed_shot, "missed", "confirmed"
        elif v.verdict == "no_shot":
            new.status = "rejected"
        else:
            new.status = "needs_review"
        if v.shot_kind != "unknown":
            shot.kind = v.shot_kind
        new.shot = shot
        new.confidence = round(min(1.0, max(0.0, float(v.confidence))), 4)
        new.confidence_kind = "self_reported"
        new.description = (
            _describe(new.type, shot.kind) if v.verdict in ("made", "missed") else new.description
        )
        new.features["vlm"] = {
            "verdict": v.verdict,
            "confidence": new.confidence,
            "shot_kind": v.shot_kind,
            "reason": v.reason,
            "evidence_tiles": list(v.evidence_tiles),
            "shooter_description": v.shooter_description,
            "agrees_with_system_one": v.verdict
            == {"made_shot": "made", "missed_shot": "missed"}.get(e.type.value),
            "before": before,
        }
        t = e.t_key_s if e.t_key_s is not None else e.t_start_s
        new.evidence += [Evidence(t_s=t, tile_id=tile, note=v.reason[:200]) for tile in v.evidence_tiles]
        new.provenance = Provenance(
            source="vlm",
            producer=producer,
            stage="adjudicate",
            params={"verdict": v.verdict, "before": before},
        )
        out.append(new)
    return out


def recompute(events: list[Event], preset: Preset) -> list[Event]:
    """Re-gate events from stored System One answers under another preset (Strict <-> Eager).

    No inference. Human-reviewed events, human-added events, and events System Two already ruled on
    keep their status.
    """
    out: list[Event] = []
    for e in events:
        s1 = e.features.get("system_one")
        if (
            not s1
            or e.review != "unreviewed"
            or e.provenance.source in ("human", "vlm")
            or "vlm" in e.features
        ):
            out.append(e)
            continue
        r = route(s1, preset)
        new = e.model_copy(deep=True)
        new.status = STATUS[r]  # type: ignore[assignment]
        new.features.update(route=r, preset=preset.name)
        new.provenance.params = {
            **new.provenance.params,
            "preset": preset.name,
            "accept": preset.accept,
            "reject": preset.reject,
        }
        out.append(new)
    return out


def triage(
    decider: Decider, state: dict[str, Any] | str, *, run_id: str = "run", min_confidence: float = 0.5
) -> tuple[FootageType, DecisionRecord]:
    """Footage triage (once per clip). Below ``min_confidence`` (or on error) the type is 'unknown'."""
    res = _safe_decide(decider, state, TRIAGE)
    ans = res.answers.get("footage_type")
    routed = gate(ans) if ans is not None else None
    ok = ans is not None and ans.value in _FOOTAGE and (ans.confidence or 0.0) >= min_confidence
    footage: FootageType = ans.value if ok else "unknown"  # type: ignore[assignment,union-attr]
    return footage, _record(run_id, "clip", "triage", res, state, TRIAGE, routed)
