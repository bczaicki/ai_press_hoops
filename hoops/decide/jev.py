"""TypeSafe Jev: the primary System One decider.

Jev takes text or JSON state only (no images, 32k context) and returns typed answers with
calibrated probabilities. Calls go through typesafe-sdk 0.7.4 (``TypeSafeClient.system_one``).
Input costs $0.042 per 1M tokens and output is free (hoops.config.PRICES).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoops.config import get_settings
from hoops.decide.base import DecisionResult, Question
from hoops.decide.common import cost_usd, error_text, estimate_tokens, from_typesafe, to_jsonable, to_typesafe
from hoops.registry import deciders

DEFAULT_MODEL = "jev-latest"


@dataclass
class JevDecider:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout_s: float = 10.0
    max_retries: int = 2
    client: Any = None  # a TypeSafeClient (or a test double); built lazily
    key: str = "jev"
    supports_images: bool = False

    def _client(self) -> Any:
        if self.client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self.client = TypeSafeClient(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout_s,
                retry=RetryPolicy(max_retries=self.max_retries),
            )
        return self.client

    def decide(
        self, state: dict[str, Any] | str, questions: dict[str, Question], images: Sequence[bytes] = ()
    ) -> DecisionResult:
        t0 = time.perf_counter()
        try:
            import typesafe_sdk as sdk

            resp = self._client().system_one(
                state=state, questions=to_typesafe(questions, sdk), model=self.model
            )
            answers = {
                n: from_typesafe(q, resp.answers[n]) for n, q in questions.items() if n in resp.answers
            }
        except Exception as exc:  # noqa: BLE001 - deciders report provider errors instead of raising
            ms = (time.perf_counter() - t0) * 1e3
            return DecisionResult({}, self.key, self.model, latency_ms=ms, error=error_text(exc))
        ms = (time.perf_counter() - t0) * 1e3
        model = getattr(resp, "model", None) or self.model
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "input_tokens", None)
        if tokens is None:
            tokens = estimate_tokens(state, questions)
        cost = cost_usd(model, tokens)
        if cost is None:  # a dated snapshot name we don't price yet: use the alias price
            cost = cost_usd(DEFAULT_MODEL, tokens)
        missing = sorted(set(questions) - set(answers))
        raw = {"response": to_jsonable(resp), "missing_answers": missing} if missing else to_jsonable(resp)
        return DecisionResult(answers, self.key, model, latency_ms=ms, cost_usd=cost, raw=raw)


@deciders.register(
    "jev",
    label="TypeSafe Jev (System One)",
    license="MIT SDK; hosted model (TypeSafe terms)",
    requires_modules=("typesafe_sdk",),
    requires_keys=("typesafe_api_key",),
    notes="Text/JSON state only; calibrated probabilities; $0.042 per 1M input tokens.",
    supports_images=False,
    default_model=DEFAULT_MODEL,
)
def make_jev(model: str = DEFAULT_MODEL, **kw: Any) -> JevDecider:
    return JevDecider(model=model, api_key=get_settings().key("typesafe_api_key"), **kw)
