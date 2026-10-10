"""Helpers shared by the System One deciders: SDK question/answer mapping, cost, errors."""

from __future__ import annotations

import json
from typing import Any

from hoops.config import PRICES
from hoops.decide.base import Answer, Choice, Noul, Question, Score, question_spec


def to_typesafe(questions: dict[str, Question], sdk: Any) -> dict[str, Any]:
    """Map our questions onto typesafe_sdk/system_one_adapter question models (``sdk`` = that module)."""
    out: dict[str, Any] = {}
    for name, q in questions.items():
        if isinstance(q, Choice):
            out[name] = sdk.Choice(instructions=q.instructions, criteria=dict(q.criteria))
        elif isinstance(q, Score):
            out[name] = sdk.Score(instructions=q.instructions, criteria=list(q.criteria))
        else:
            out[name] = sdk.Noul(instructions=q.instructions)
    return out


def from_typesafe(q: Question, a: Any) -> Answer:
    """Map a typesafe answer (NoulAnswer | ChoiceAnswer | ScoreAnswer) onto our Answer."""
    if isinstance(q, Noul):
        return Answer("noul", float(a.noul))
    if isinstance(q, Choice):
        probs = {str(k): float(v) for k, v in a.probabilities.items()}
        return Answer("choice", str(a.choice), probabilities=probs, confidence=float(a.confidence))
    levels = list(q.criteria)
    probs = {levels[int(k)]: float(v) for k, v in a.probabilities.items() if 0 <= int(k) < len(levels)}
    level = (
        max(probs, key=probs.__getitem__) if probs else levels[min(len(levels) - 1, max(0, round(a.score)))]
    )
    return Answer(
        "score",
        level,
        probabilities=probs,
        confidence=float(a.confidence),
        raw={"expected_score": float(a.score)},
    )


def estimate_tokens(state: Any, questions: dict[str, Question]) -> int:
    """Rough input-token estimate (chars / 4) when a provider doesn't report usage."""
    text = state if isinstance(state, str) else json.dumps(state, default=str)
    specs = json.dumps({k: question_spec(q) for k, q in questions.items()})
    return (len(text) + len(specs)) // 4


def cost_usd(model: str | None, input_tokens: int | None, output_tokens: int | None = 0) -> float | None:
    """USD from hoops.config.PRICES (per 1M tokens), or None for an unknown model."""
    if model is None or model not in PRICES or input_tokens is None:
        return None
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + (output_tokens or 0) * p_out) / 1e6


def error_text(exc: BaseException) -> str:
    """Short error string for DecisionResult.error (SDK errors redact auth headers)."""
    msg = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {msg[:300]}"


def to_jsonable(obj: Any) -> Any:
    """Best-effort JSON-friendly dump of an SDK response for DecisionResult.raw."""
    for attr in ("model_dump",):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(mode="json")
            except Exception:  # noqa: BLE001 - raw is diagnostic only
                break
    return repr(obj)[:2000]
