"""Provider billing health: turn 'out of credits' errors into a fix-it message, and probe at startup.

Anthropic's token counting is free and reports an empty balance. OpenAI has no free billing check, so its probe
is one ~10-token call on the cheapest model (about $0.00001). HOOPS_HEALTH_PROBES=0 disables network probes.
"""

from __future__ import annotations

import os

from hoops.config import PROFILES, get_settings

BILLING_PATTERNS = (
    "credit balance",
    "insufficient_quota",
    "no credits",
    "credit_balance",
    "billing",
    "quota",
)
CONSOLES = {
    "claude": "console.anthropic.com",
    "openai": "platform.openai.com",
    "gemini": "aistudio.google.com",
    "openrouter": "openrouter.ai",
}


def billing_problem(provider: str, error: str) -> str | None:
    """A friendly reason when a provider error means 'out of credits / no billing', else None."""
    if any(p in error.lower() for p in BILLING_PATTERNS):
        where = CONSOLES.get(provider, "the provider's console")
        label = PROFILES[provider].label if provider in PROFILES else provider
        return f"{label} account is out of credits (add billing at {where})"
    return None


def probes_enabled() -> bool:
    return os.environ.get("HOOPS_HEALTH_PROBES", "1") != "0"


def probe_billing() -> dict[str, str]:
    """provider key -> reason, for providers with a key whose account can't be used right now."""
    if not probes_enabled():
        return {}
    settings = get_settings()
    problems: dict[str, str] = {}
    if key := settings.key("anthropic_api_key"):
        try:
            import anthropic

            anthropic.Anthropic(api_key=key, max_retries=0, timeout=15).messages.count_tokens(
                model=PROFILES["claude"].default_model, messages=[{"role": "user", "content": "hi"}]
            )
        except Exception as exc:  # noqa: BLE001 - any failure that isn't billing is reported at use time
            if why := billing_problem("claude", str(exc)):
                problems["claude"] = why
    if key := settings.key("openai_api_key"):
        try:
            import openai

            openai.OpenAI(api_key=key, max_retries=0, timeout=15).responses.create(
                model="gpt-6-luna", input="Reply with OK.", max_output_tokens=16
            )
        except Exception as exc:  # noqa: BLE001
            if why := billing_problem("openai", str(exc)):
                problems["openai"] = why
    return problems
