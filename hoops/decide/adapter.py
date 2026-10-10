"""LLM-backed System One via ``system-one-adapter`` (same ``system_one()`` shape as Jev).

Before there is Jev access, Claude, OpenAI, or Gemini answer the same typed questions. The adapter
asks for a probability per option and normalizes them. Its confidences come from the LLM's own
probabilities, so they are not calibrated the way Jev's are.

Claude notes (claude-opus-5-5): thinking is always on, and thinking tokens count against
``max_tokens``, so the adapter's 4096 default can cut a response short. We pass our own provider
instance with ``max_tokens=16000`` and ``output_config.effort="low"`` (Opus 5.5 defaults to
medium), plus server-side refusal fallbacks (``fallbacks="default"``, beta
``server-side-fallback-2026-07-01``). We never send temperature, disabled thinking, or a prefill.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoops.config import get_settings
from hoops.decide.base import DecisionResult, Question
from hoops.decide.common import cost_usd, error_text, from_typesafe, to_jsonable, to_typesafe
from hoops.registry import deciders

# registry suffix -> (adapter provider, settings key field, default model, SDK module)
PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "claude": ("anthropic", "anthropic_api_key", "claude-opus-5-5", "anthropic"),
    "openai": ("openai", "openai_api_key", "gpt-6.1-sol", "openai"),
    "gemini": ("gemini", "gemini_api_key", "gemini-3.8-flash", "google.genai"),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def claude_provider(
    model: str,
    *,
    api_key: str | None,
    max_tokens: int = 16000,
    effort: str | None = "low",
    fallbacks: bool = True,
) -> Any:
    """An adapter SyncProvider for Claude with an explicit key, max_tokens, effort, and fallbacks."""
    import anthropic
    from system_one_adapter.providers import anthropic as base
    from system_one_adapter.providers.base import record_request, translating

    class ClaudeProvider(base.AnthropicProvider):
        def __init__(self) -> None:
            self.model_name, self.max_tokens = model, max_tokens
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)

        def request(self, messages: list[Any], *, schema: dict[str, Any], structured: bool) -> Any:
            with translating(self.translate_error):
                kwargs = base._request_kwargs(
                    self.model_name, messages, schema, structured=structured, max_tokens=self.max_tokens
                )
                if effort:
                    kwargs.setdefault("output_config", {})["effort"] = effort
                if fallbacks:
                    kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
                record_request(kwargs, api="messages")
                messages_api = self._client.beta.messages if fallbacks else self._client.messages
                response = messages_api.create(**kwargs)
            return base._result(response)

    return ClaudeProvider()


@dataclass
class AdapterDecider:
    provider: str  # "anthropic" | "openai" | "gemini"
    model: str
    key: str  # registry key, e.g. "adapter:claude"
    api_key: str | None = None
    max_tokens: int = 16000  # Claude only
    effort: str | None = "low"  # Claude only
    fallbacks: bool = True  # Claude only
    retries: int = 2
    client: Any = None  # SystemOneAdapterClient (or a test double); built lazily
    provider_instance: Any = None  # adapter SyncProvider; built lazily
    supports_images: bool = False

    def _provider(self) -> Any:
        if self.provider_instance is None:
            if self.provider == "anthropic":
                self.provider_instance = claude_provider(
                    self.model,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    effort=self.effort,
                    fallbacks=self.fallbacks,
                )
            elif self.provider == "openai":
                from system_one_adapter.providers.openai import OpenAIProvider

                self.provider_instance = OpenAIProvider(self.model, api_key=self.api_key)
            elif self.provider == "gemini":
                from system_one_adapter.providers.gemini import GeminiProvider

                self.provider_instance = GeminiProvider(self.model, api_key=self.api_key)
            else:
                raise ValueError(f"unknown adapter provider {self.provider!r}")
        return self.provider_instance

    def _client(self) -> Any:
        if self.client is None:
            from system_one_adapter import RetryPolicy, SystemOneAdapterClient

            self.client = SystemOneAdapterClient(
                structured_outputs=True,
                llm_answer_mode="probabilities",
                normalize_probabilities=True,
                n_retry_malformed_structure=1,
                retry=RetryPolicy(max_retries=self.retries),
            )
        return self.client

    def decide(
        self, state: dict[str, Any] | str, questions: dict[str, Question], images: Sequence[bytes] = ()
    ) -> DecisionResult:
        t0 = time.perf_counter()
        try:
            import system_one_adapter as sdk

            resp = self._client().system_one(
                state=state, questions=to_typesafe(questions, sdk), model=self._provider()
            )
            answers = {
                n: from_typesafe(q, resp.answers[n]) for n, q in questions.items() if n in resp.answers
            }
        except Exception as exc:  # noqa: BLE001 - deciders report provider errors instead of raising
            ms = (time.perf_counter() - t0) * 1e3
            return DecisionResult({}, self.key, self.model, latency_ms=ms, error=error_text(exc))
        ms = (time.perf_counter() - t0) * 1e3
        usage = getattr(resp, "usage", None)
        t_in = getattr(usage, "input_tokens_total", None) or getattr(usage, "input_tokens", None)
        t_out = getattr(usage, "output_tokens_total", None) or getattr(usage, "output_tokens", None)
        raw = to_jsonable(resp)
        if isinstance(raw, dict):
            raw.pop("debug", None)  # full prompts/attempt traces: large, and not needed in run.json
        return DecisionResult(
            answers, self.key, self.model, latency_ms=ms, cost_usd=cost_usd(self.model, t_in, t_out), raw=raw
        )


def _register(suffix: str, provider: str, key_field: str, default_model: str, module: str) -> None:
    label = {"claude": "Claude", "openai": "OpenAI", "gemini": "Gemini"}[suffix]

    @deciders.register(
        f"adapter:{suffix}",
        label=f"{label} as System One (system-one-adapter)",
        license="MIT (system-one-adapter)",
        requires_modules=("system_one_adapter", module),
        requires_keys=(key_field,),
        notes="LLM-backed stand-in for Jev; text state only; confidences are not calibrated.",
        supports_images=False,
        default_model=default_model,
    )
    def make(model: str = default_model, **kw: Any) -> AdapterDecider:
        return AdapterDecider(
            provider=provider,
            model=model,
            key=f"adapter:{suffix}",
            api_key=get_settings().key(key_field),
            **kw,
        )


for _suffix, (_provider, _key_field, _model, _module) in PROVIDERS.items():
    _register(_suffix, _provider, _key_field, _model, _module)
