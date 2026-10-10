"""Settings, provider profiles, model defaults, and prices.

Model IDs and prices live here (config), not in code: they change every few weeks.
Prices are USD per 1M tokens, checked 2026-10-10 against provider model lists.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Provider keys. All optional: a missing key disables that provider in the UI.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    openrouter_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None
    roboflow_api_key: SecretStr | None = None

    # Gradio auth for share links (the link is public; auth is not optional when sharing).
    app_user: str = Field("friend", validation_alias="HOOPS_APP_USER")
    app_password: SecretStr | None = Field(
        None, validation_alias=AliasChoices("HOOPS_APP_PASSWORD", "APP_PASSWORD")
    )

    data_dir: Path = Field(ROOT / "data", validation_alias="HOOPS_DATA_DIR")
    ffmpeg_bin: str = Field("ffmpeg", validation_alias="HOOPS_FFMPEG")
    ffprobe_bin: str = Field("ffprobe", validation_alias="HOOPS_FFPROBE")

    # Guardrails
    run_cost_cap_usd: float = Field(0.50, validation_alias="HOOPS_RUN_COST_CAP_USD")
    session_cost_cap_usd: float = Field(5.00, validation_alias="HOOPS_SESSION_COST_CAP_USD")
    max_upload_mb: int = Field(500, validation_alias="HOOPS_MAX_UPLOAD_MB")
    max_clip_seconds: float = Field(600.0, validation_alias="HOOPS_MAX_CLIP_SECONDS")

    # Defaults (registry keys)
    default_detector: str = Field("rfdetr:coco", validation_alias="HOOPS_DETECTOR")
    default_tracker: str = Field("bytetrack", validation_alias="HOOPS_TRACKER")
    default_decider: str = Field("rules", validation_alias="HOOPS_DECIDER")
    default_vlm: str = Field("claude", validation_alias="HOOPS_VLM")

    ollama_host: str = Field("http://localhost:11434", validation_alias="OLLAMA_HOST")
    ollama_model: str | None = Field(None, validation_alias="HOOPS_OLLAMA_MODEL")  # else profile default
    mlx_vlm_url: str = Field("http://localhost:8080/v1", validation_alias="HOOPS_MLX_VLM_URL")

    def key(self, name: str) -> str | None:
        """Return a provider key by settings field name (e.g. 'anthropic_api_key'), or None."""
        value = getattr(self, name, None)
        return value.get_secret_value() if isinstance(value, SecretStr) and value.get_secret_value() else None


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  (aliased fields have defaults)


class ProviderProfile(BaseModel):
    """A fixed VLM provider profile. Users pick a profile; they never type a base URL (SSRF/key misuse)."""

    key: str
    label: str
    kind: Literal["anthropic", "gemini", "openai_compat", "ollama"]
    default_model: str
    key_field: str | None  # Settings field holding the API key; None for local servers
    base_url: str | None = None
    native_video: bool = False
    audio: bool = False
    max_images: int = 20
    max_px: int = 1568
    json_schema: Literal["strict", "loose", "none"] = "strict"
    local: bool = False
    sends_data_to: str  # shown in the privacy opt-in


PROFILES: dict[str, ProviderProfile] = {
    "claude": ProviderProfile(
        key="claude",
        label="Claude (Anthropic)",
        kind="anthropic",
        default_model="claude-opus-5-5",
        key_field="anthropic_api_key",
        max_images=20,
        max_px=2576,
        sends_data_to="Anthropic",
    ),
    "gemini": ProviderProfile(
        key="gemini",
        label="Gemini (Google, native video)",
        kind="gemini",
        default_model="gemini-3.8-flash",
        key_field="gemini_api_key",
        native_video=True,
        audio=True,
        max_images=500,
        max_px=3072,
        sends_data_to="Google (paid tier only)",
    ),
    "openai": ProviderProfile(
        key="openai",
        label="OpenAI",
        kind="openai_compat",
        default_model="gpt-6.1-sol",
        key_field="openai_api_key",
        base_url=None,
        max_images=100,
        max_px=2048,
        sends_data_to="OpenAI",
    ),
    "openrouter": ProviderProfile(
        key="openrouter",
        label="OpenRouter (any model)",
        kind="openai_compat",
        default_model="google/gemini-3.8-flash",
        key_field="openrouter_api_key",
        base_url="https://openrouter.ai/api/v1",
        max_images=20,
        max_px=2048,
        json_schema="loose",
        sends_data_to="OpenRouter + the routed model provider",
    ),
    "ollama": ProviderProfile(
        key="ollama",
        label="Local (Ollama)",
        kind="ollama",
        default_model="qwen3.5:4b",
        key_field=None,
        max_images=8,
        max_px=1024,
        local=True,
        sends_data_to="nobody (runs on this Mac)",
    ),
}

# USD per 1M tokens: (input, output). Prompt-length tiers noted inline.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts <=100k tokens; 0.50/2.50 above
    "gemini-3.8-flash": (0.75, 3.75),  # through 2026-12-31, then 1.50/7.50
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-astra": (10.00, 50.00),
    "jev-latest": (0.042, 0.0),
    "jev-1.13.0": (0.042, 0.0),
    # Older/fallback models that can answer (e.g. Opus 5.5's server-side refusal fallback targets)
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-6-sol": (2.00, 10.00),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
}

OUTPUTS = ("annotated_video", "events", "summary", "court_map")
