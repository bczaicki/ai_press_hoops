"""System One contracts: fast, typed, calibrated decisions.

Question types mirror TypeSafe's ``typesafe_sdk`` (Choice | Score | Noul) so Jev, the LLM-backed
``system-one-adapter``, OpenAI's Decisions API, and our rules baseline are interchangeable.

What each answer type returns (per docs.typesafe.ai):
- Choice: the chosen option + probabilities over options + a confidence.
- Score:  the chosen ordered level + probabilities + a confidence.
- Noul:   ONLY a probability that the statement is true (no separate confidence).

State given to deciders is short, relevant, qualitative text/JSON written by code. Never ask a
System One model to count, do arithmetic, or compare numbers/times (Jev 1.13 can't); do that in code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: dict[str, str | None]  # option -> description (max 255 options)


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[str]  # ordered levels, lowest first (max 10)


@dataclass(frozen=True)
class Noul:
    instructions: str


Question = Choice | Score | Noul


def question_spec(q: Question) -> dict[str, Any]:
    """Serializable form for DecisionRecord.questions."""
    return {"kind": type(q).__name__.lower(), **asdict(q)}


@dataclass
class Answer:
    kind: Literal["choice", "score", "noul"]
    value: str | float  # Choice option | Score level | Noul probability in [0, 1]
    probabilities: dict[str, float] | None = None
    confidence: float | None = None  # Choice/Score only; None for Noul
    raw: Any = None

    def spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class DecisionResult:
    answers: dict[str, Answer]
    decider: str  # registry key, e.g. "jev"
    model: str | None = None  # e.g. "jev-latest", "claude-opus-5-5"
    latency_ms: float = 0.0
    cost_usd: float | None = None
    error: str | None = None  # set (and answers empty) when the call failed
    raw: Any = None


@runtime_checkable
class Decider(Protocol):
    key: str
    supports_images: bool

    def decide(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, Question],
        images: Sequence[bytes] = (),
    ) -> DecisionResult: ...


Route = Literal["accept", "reject", "escalate"]


def gate(answer: Answer, accept: float = 0.85, reject: float = 0.15) -> Route:
    """Confidence-gated routing (TypeSafe 'confidence routing' pattern).

    - Noul: accept if p >= accept, reject if p <= reject, otherwise escalate to System Two.
    - Choice/Score: accept if confidence >= accept, otherwise escalate (there is no 'reject' for a
      categorical answer; the chosen option itself carries the decision).
    """
    if answer.kind == "noul":
        p = float(answer.value)
        if p >= accept:
            return "accept"
        if p <= reject:
            return "reject"
        return "escalate"
    conf = answer.confidence
    if conf is not None and conf >= accept:
        return "accept"
    return "escalate"


@dataclass(frozen=True)
class Preset:
    """Strict <-> Eager presets (Type I vs Type II error trade-off)."""

    name: str
    accept: float
    reject: float
    note: str = ""


PRESETS: dict[str, Preset] = {
    "strict": Preset("strict", accept=0.92, reject=0.30, note="fewer phantom makes; more go to review"),
    "balanced": Preset("balanced", accept=0.85, reject=0.15),
    "eager": Preset("eager", accept=0.70, reject=0.05, note="miss fewer shots; more false alarms"),
}


@dataclass
class DeciderInfo:
    key: str
    model: str | None
    supports_images: bool
    notes: list[str] = field(default_factory=list)
