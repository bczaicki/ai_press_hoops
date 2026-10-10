"""OpenAI Decisions API (public beta) as a System One decider. It accepts images.

``client.decisions.create(model, input, questions)`` (openai 3.28). Question types map like this:
predicate is Noul, choice is Choice, and score is Score. ``input`` is a string, or user messages with
``input_text`` and inline ``input_image`` parts (base64 data URLs only, at most 128 images). Answers
come back in question order. Any answer may be a ``refusal``, which we leave out and record in raw.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoops.config import get_settings
from hoops.decide.base import Answer, Choice, DecisionResult, Noul, Question, Score
from hoops.decide.common import cost_usd, error_text, to_jsonable
from hoops.registry import deciders

DEFAULT_MODEL = "gpt-6-luna"


def data_url(img: bytes) -> str:
    mime = "image/png" if img[:4] == b"\x89PNG" else "image/webp" if img[8:12] == b"WEBP" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(img).decode()}"


def to_openai_questions(questions: dict[str, Question]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, q in questions.items():
        if isinstance(q, Noul):
            out.append({"type": "predicate", "name": name, "instructions": q.instructions})
        elif isinstance(q, Choice):
            choices = [{"value": k, "description": v} if v else {"value": k} for k, v in q.criteria.items()]
            out.append({"type": "choice", "name": name, "instructions": q.instructions, "choices": choices})
        elif isinstance(q, Score):
            levels = [{"label": level} for level in q.criteria]
            out.append({"type": "score", "name": name, "instructions": q.instructions, "levels": levels})
    return out


def to_openai_input(state: dict[str, Any] | str, images: Sequence[bytes], detail: str) -> Any:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if not images:
        return text
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    parts += [{"type": "input_image", "image_url": data_url(img), "detail": detail} for img in images]
    return [{"role": "user", "content": parts}]


def from_openai_answers(
    questions: dict[str, Question], answers: Sequence[Any]
) -> tuple[dict[str, Answer], list[str]]:
    names = list(questions)
    out: dict[str, Answer] = {}
    refused: list[str] = []
    for i, a in enumerate(answers):
        name = getattr(a, "name", None) or (names[i] if i < len(names) else None)
        if name not in questions:
            continue
        kind = getattr(a, "type", None)
        if kind == "refusal":
            refused.append(name)
        elif kind == "predicate":
            out[name] = Answer("noul", float(a.probability))
        elif kind == "choice":
            probs = {str(p.value): float(p.probability) for p in a.probabilities}
            out[name] = Answer("choice", str(a.choice), probabilities=probs, confidence=float(a.confidence))
        elif kind == "score":
            probs = {str(p.label): float(p.probability) for p in a.probabilities}
            level = max(probs, key=probs.__getitem__) if probs else ""
            out[name] = Answer(
                "score",
                level,
                probabilities=probs,
                confidence=float(a.confidence),
                raw={"expected_score": float(a.score)},
            )
    return out, refused


@dataclass
class OpenAIDecisionsDecider:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    image_detail: str = "low"
    max_images: int = 16
    timeout_s: float = 30.0
    client: Any = None  # openai.OpenAI (or a test double); built lazily
    key: str = "openai-decisions"
    supports_images: bool = True

    def _client(self) -> Any:
        if self.client is None:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout_s, max_retries=2)
        return self.client

    def decide(
        self, state: dict[str, Any] | str, questions: dict[str, Question], images: Sequence[bytes] = ()
    ) -> DecisionResult:
        t0 = time.perf_counter()
        try:
            resp = self._client().decisions.create(
                model=self.model,
                input=to_openai_input(state, list(images)[: self.max_images], self.image_detail),
                questions=to_openai_questions(questions),
            )
            answers, refused = from_openai_answers(questions, resp.answers)
        except Exception as exc:  # noqa: BLE001 - deciders report provider errors instead of raising
            ms = (time.perf_counter() - t0) * 1e3
            return DecisionResult({}, self.key, self.model, latency_ms=ms, error=error_text(exc))
        ms = (time.perf_counter() - t0) * 1e3
        usage = getattr(resp, "usage", None)
        model = getattr(resp, "model", None) or self.model
        price_model = model if cost_usd(model, 0) is not None else self.model
        cost = cost_usd(
            price_model, getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
        )
        raw = to_jsonable(resp)
        error = f"refused: {', '.join(refused)}" if refused and not answers else None
        return DecisionResult(answers, self.key, model, latency_ms=ms, cost_usd=cost, error=error, raw=raw)


@deciders.register(
    "openai-decisions",
    label="OpenAI Decisions (gpt-6-luna, sees images)",
    license="Apache-2.0 SDK; hosted model (OpenAI terms)",
    requires_modules=("openai",),
    requires_keys=("openai_api_key",),
    notes="Public beta. Accepts inline images (e.g. rim crops) for predicates like 'ball went through the hoop'.",
    supports_images=True,
    default_model=DEFAULT_MODEL,
)
def make_openai_decisions(model: str = DEFAULT_MODEL, **kw: Any) -> OpenAIDecisionsDecider:
    return OpenAIDecisionsDecider(model=model, api_key=get_settings().key("openai_api_key"), **kw)
