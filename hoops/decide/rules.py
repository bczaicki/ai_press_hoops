"""Deterministic System One baseline: known questions answered from rule scores and descriptors.

Free, instant, and always available, so it is the fallback when a hosted decider fails.
Probabilities come from the shot state machine's heuristic scores (``event_state(...,
with_scores=True)``) or, without them, from the categorical descriptors. A Choice's confidence is
the probability of the chosen option. Unknown questions get an uninformative answer (Noul 0.5,
uniform Choice/Score with confidence 0) so they always escalate.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoops.decide.base import Answer, Choice, DecisionResult, Noul, Question, Score
from hoops.registry import deciders

_ROUTE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stats", ("how many", "how much", "percent", "%", "stat", "points", "total", "count", "shooting", "fg")),
    (
        "find_moments",
        ("show me", "when ", "where", "find", "which moment", "timestamp", "every time", "clip of"),
    ),
    ("explain_play", ("why", "explain", "how did", "what happened on", "what went wrong", "break down")),
    ("summary", ("summar", "recap", "overview", "what happened", "tell me about", "highlight")),
)


def uninformative(q: Question) -> Answer:
    if isinstance(q, Noul):
        return Answer("noul", 0.5)
    options = list(q.criteria)
    probs = dict.fromkeys(options, 1 / len(options)) if options else {}
    if isinstance(q, Score):
        return Answer(
            "score", options[len(options) // 2] if options else "", probabilities=probs, confidence=0.0
        )
    return Answer("choice", options[0] if options else "", probabilities=probs, confidence=0.0)


def _choice(q: Choice, weights: dict[str, float]) -> Answer:
    """Normalize weights over the question's options; value = argmax, confidence = its probability."""
    probs = {opt: max(0.0, float(weights.get(opt, 0.0))) for opt in q.criteria}
    total = sum(probs.values())
    if total <= 0:
        return uninformative(q)
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    best = max(probs, key=probs.__getitem__)
    return Answer("choice", best, probabilities=probs, confidence=probs[best])


def _get(state: dict[str, Any], key: str) -> str:
    return str(state.get(key, "")).lower()


def _attempt_p(s: dict[str, Any]) -> float:
    scores = s.get("rule_scores") or {}
    if "attempt" in scores:
        return float(scores["attempt"])
    path = _get(s, "ball_path")
    p = 0.8 if ("rose above rim" in path or "bounced" in path) else 0.6 if "detector fired" in path else 0.5
    if _get(s, "ball_visible") == "few frames":
        p -= 0.15
    if _get(s, "scene_cut_nearby") == "yes":
        p -= 0.2
    return min(0.98, max(0.02, p))


def _make_p(s: dict[str, Any]) -> float | None:
    """P(made | attempt), or None when the evidence can't say."""
    scores = s.get("rule_scores") or {}
    if "make" in scores:
        return float(scores["make"])
    crossing, bib = _get(s, "crossing"), _get(s, "ball_in_basket_class")
    base = {"inside rim edges": 0.9, "at the rim edge": 0.4, "outside rim edges": 0.08}.get(crossing)
    if "several" in bib:
        return max(base or 0.0, 0.92)
    if base is None:
        return 0.75 if "briefly" in bib else None
    return base


def _outcome(q: Choice, s: dict[str, Any]) -> Answer:
    m = _make_p(s)
    if m is None:
        return _choice(q, {"made": 0.25, "missed": 0.25, "unclear": 0.5})
    return _choice(q, {"made": m, "missed": 1 - m, "unclear": 0.0})


def _shot_kind(q: Choice, s: dict[str, Any]) -> Answer:
    motion, dist, arc = _get(s, "shooter_motion"), _get(s, "shooter_distance"), _get(s, "arc")
    if "driving" in motion:
        return _choice(q, {"layup": 0.6, "dunk": 0.25, "unknown": 0.15})
    if "close" in dist:
        w = (
            {"layup": 0.5, "dunk": 0.15, "tip": 0.1, "unknown": 0.25}
            if "just above" in arc
            else {"layup": 0.35, "jumper": 0.3, "unknown": 0.35}
        )
        return _choice(q, w)
    if dist in ("mid-range", "far from the rim") or "jumping" in motion or "high above" in arc:
        return _choice(q, {"jumper": 0.7, "free_throw": 0.05, "unknown": 0.25})
    return _choice(q, {"unknown": 0.6, "jumper": 0.2, "layup": 0.2})


def _footage(q: Choice, s: dict[str, Any]) -> Answer:
    w = dict.fromkeys(q.criteria, 0.05)
    cuts, motion, rim = _get(s, "scene_cuts"), _get(s, "camera_motion"), _get(s, "rim")
    players, board = _get(s, "players"), _get(s, "scoreboard")
    if "not seen" in rim and players in ("none", "not seen", ""):
        w["not_basketball"] = w.get("not_basketball", 0) + 0.6
    if board == "visible":
        w["broadcast"] = w.get("broadcast", 0) + 0.5
    if cuts == "many":
        w["highlights"] = w.get("highlights", 0) + 0.3
        w["broadcast"] = w.get("broadcast", 0) + 0.2
    if motion in ("steady", "static", "none"):
        w["fixed_camera"] = w.get("fixed_camera", 0) + 0.5
    elif motion in ("shaky", "handheld"):
        w["handheld"] = w.get("handheld", 0) + 0.5
    elif motion == "panning":
        w["broadcast"] = w.get("broadcast", 0) + 0.2
        w["handheld"] = w.get("handheld", 0) + 0.2
    if players == "few":
        w["practice"] = w.get("practice", 0) + 0.3
    return _choice(q, w)


def _route(q: Choice, s: dict[str, Any]) -> Answer:
    text = f" {_get(s, 'question')} "
    hits = {route: sum(word in text for word in words) for route, words in _ROUTE_WORDS}
    best = max(hits, key=hits.__getitem__)
    if hits[best] == 0:
        return _choice(
            q,
            {"unsupported": 0.4, "summary": 0.15, "stats": 0.15, "find_moments": 0.15, "explain_play": 0.15},
        )
    top = 0.85 if hits[best] >= 2 else 0.75
    rest = (1 - top) / max(1, len(q.criteria) - 1)
    return _choice(q, {opt: (top if opt == best else rest) for opt in q.criteria})


def _court_lines(s: dict[str, Any]) -> Answer:
    v = _get(s, "court_lines")
    return Answer("noul", 0.2 if "not" in v else 0.8 if "visible" in v or "clear" in v else 0.5)


@dataclass
class RulesDecider:
    key: str = "rules"
    supports_images: bool = False
    wants_scores: bool = True  # ask for event_state(..., with_scores=True)
    model: str = "rules-v1"

    def answer(self, name: str, q: Question, s: dict[str, Any]) -> Answer:
        if name == "is_shot_attempt" and isinstance(q, Noul):
            return Answer("noul", round(_attempt_p(s), 4))
        if name == "outcome" and isinstance(q, Choice):
            return _outcome(q, s)
        if name == "shot_kind" and isinstance(q, Choice):
            return _shot_kind(q, s)
        if name == "footage_type" and isinstance(q, Choice):
            return _footage(q, s)
        if name == "court_lines_visible" and isinstance(q, Noul):
            return _court_lines(s)
        if name == "route" and isinstance(q, Choice):
            return _route(q, s)
        return uninformative(q)  # includes claim_check "supported": rules can't verify prose

    def decide(
        self, state: dict[str, Any] | str, questions: dict[str, Question], images: Sequence[bytes] = ()
    ) -> DecisionResult:
        t0 = time.perf_counter()
        s = state if isinstance(state, dict) else {"question": state, "text": state}
        answers = {name: self.answer(name, q, s) for name, q in questions.items()}
        return DecisionResult(
            answers=answers,
            decider=self.key,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            cost_usd=0.0,
        )


@deciders.register(
    "rules",
    label="Rules (baseline)",
    license="",
    notes="Deterministic heuristics over the shot state machine's scores; free and instant.",
    supports_images=False,
    default_model="rules-v1",
)
def make_rules(**_: Any) -> RulesDecider:
    return RulesDecider()
