"""Token estimators, USD prices, and run/session spend caps.

Usage convention (every adapter normalizes to it): ``Usage.input_tokens`` counts *uncached* input tokens only;
cache reads and cache writes are reported separately; ``output_tokens`` includes thinking/reasoning tokens
(all providers bill them as output). Estimates are deliberately a little pessimistic: a cap that trips early
is better than a surprise bill.
"""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Iterable

from PIL import Image

from hoops.config import PRICES, PROFILES, get_settings
from hoops.reason.base import AnalyzeRequest, Usage

# Price multipliers relative to the input price.
CACHE_READ_FACTOR = {"claude-opus-5-5": 0.05, "gpt-6.1-sol": 0.05}  # default 0.10
CACHE_WRITE_FACTOR = 1.25  # Anthropic 5-minute cache writes; other providers have no write premium
# Long-prompt tiers: (threshold input tokens, multiplier on both prices).
LONG_PROMPT_TIERS = {
    "claude-haiku-5-5": (100_000, 5.0),
    "gpt-6.1-sol": (272_000, 2.0),
    "gpt-6-luna": (272_000, 2.0),
}
# Thinking/reasoning allowance (output tokens) by effort, added to the visible-output estimate.
THINKING_ALLOWANCE = {"low": 1_000, "medium": 3_000, "high": 6_000, "xhigh": 10_000, "max": 16_000}
VISIBLE_OUTPUT_ESTIMATE = 1_500
UNKNOWN_PRICE = (2.00, 10.00)  # unknown hosted model: assume a mid-tier price rather than free


def _kind(provider_kind: str, model: str = "") -> str:
    """Normalize a profile key / profile kind / OpenRouter slug to a tokenizer family."""
    k = provider_kind.lower()
    if k in PROFILES:
        prof = PROFILES[k]
        if k == "openrouter":
            slug = model.lower()
            if slug.startswith("anthropic/"):
                return "anthropic"
            if slug.startswith("google/"):
                return "gemini"
            return "openai"
        k = prof.kind
    if k in ("claude", "anthropic"):
        return "anthropic"
    if k in ("openai", "openai_compat", "openrouter"):
        return "openai"
    return k


def image_tokens(provider_kind: str, w: int, h: int) -> int:
    """Approximate input tokens for one image of ``w`` x ``h`` pixels."""
    kind = _kind(provider_kind)
    if kind == "anthropic":
        return math.ceil(w / 28) * math.ceil(h / 28)
    if kind == "openai":  # 32-px patches x ~1.2 ("high" detail caps at 2,500 patches)
        patches = min(math.ceil(w / 32) * math.ceil(h / 32), 2_500)
        return math.ceil(patches * 1.2)
    if kind == "gemini":
        return 258
    if kind == "ollama":
        return 0
    return math.ceil(w / 28) * math.ceil(h / 28)


def video_tokens_gemini(seconds: float, fps: float = 1.0, high_res: bool = False, audio: bool = True) -> int:
    """Gemini native video: ~70 (default) or ~280 (high res) tokens per sampled frame + ~25 tokens/s audio."""
    frames = max(1, math.ceil(max(seconds, 0.0) * max(fps, 0.01)))
    total = frames * (280 if high_res else 70)
    if audio:
        total += math.ceil(25 * max(seconds, 0.0))
    return total


def text_tokens(text: str) -> int:
    return math.ceil(len(text or "") / 4)


def base_model(model: str) -> str:
    """'google/gemini-3.8-flash:free' -> 'gemini-3.8-flash' (OpenRouter slugs); other ids pass through."""
    m = (model or "").lstrip("~")
    return m.split("/", 1)[1].split(":")[0] if "/" in m else m


# Conservative prices for models missing from config.PRICES (e.g. a refusal fallback answered by another model,
# or an OpenRouter slug): never price an unknown model at $0, or the spend caps can't trip.
FAMILY_PRICES: list[tuple[str, tuple[float, float]]] = [
    ("claude-fable", (10.0, 50.0)),
    ("claude-mythos", (10.0, 50.0)),
    ("claude-opus", (5.0, 25.0)),
    ("claude-sonnet", (3.0, 15.0)),
    ("claude-haiku", (1.0, 5.0)),
    ("gpt-6-astra", (10.0, 50.0)),
    ("gpt", (5.0, 30.0)),
    ("gemini", (2.0, 12.0)),
    ("jev", (0.042, 0.0)),
]
UNKNOWN_PRICE = (10.0, 50.0)


def price_for(model: str) -> tuple[float, float] | None:
    """USD per 1M (input, output) tokens, or None when unknown."""
    return PRICES.get(base_model(model))


def price_or_conservative(model: str) -> tuple[float, float]:
    """Known price, else a conservative family price, else the most expensive tier."""
    known = price_for(model)
    if known is not None:
        return known
    base = base_model(model).lower()
    for prefix, price in FAMILY_PRICES:
        if base.startswith(prefix):
            return price
    return UNKNOWN_PRICE


def _image_dims(jpeg: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            return img.size
    except Exception:  # noqa: BLE001 - estimation must never fail
        return (1280, 720)


def _video_seconds(req: AnalyzeRequest) -> float:
    v = req.video
    if v is None:
        return 0.0
    if v.t0_s is not None and v.t1_s is not None:
        return max(0.0, v.t1_s - v.t0_s)
    try:
        import av

        with av.open(str(v.path)) as c:
            dur = (c.duration or 0) / av.time_base
        return max(0.0, dur - (v.t0_s or 0.0))
    except Exception:  # noqa: BLE001
        return 60.0


def estimate_tokens(req: AnalyzeRequest, profile_key: str, model: str) -> tuple[int, int]:
    """(input_tokens, output_tokens) expected for ``req`` on ``profile_key``/``model``."""
    kind = _kind(profile_key, model)
    text = "\n".join([req.prompt or "", req.system or "", req.cv_context or ""])
    text += "\n".join(q + "\n" + a for q, a in req.history)
    tin = text_tokens(text) + 50
    for img in req.images:
        w, h = _image_dims(img.jpeg)
        tin += image_tokens(kind, w, h) + text_tokens(img.label)
    profile = PROFILES.get(profile_key)
    if req.video is not None and profile is not None and profile.native_video:
        tin += video_tokens_gemini(_video_seconds(req), fps=req.video.fps or 1.0, audio=profile.audio)
    if req.output_model is not None:
        tin += text_tokens(str(req.output_model.model_json_schema()))
    tout = min(req.max_tokens, VISIBLE_OUTPUT_ESTIMATE)
    if kind in ("anthropic", "gemini", "openai"):  # all three bill thinking/reasoning as output
        tout = min(req.max_tokens, tout + THINKING_ALLOWANCE.get(req.effort, 3_000))
    return tin, tout


def _tiered(model: str, input_tokens: int, price: tuple[float, float]) -> tuple[float, float]:
    tier = LONG_PROMPT_TIERS.get(base_model(model))
    if tier and input_tokens > tier[0]:
        return price[0] * tier[1], price[1] * tier[1]
    return price


def estimate(req: AnalyzeRequest, profile_key: str, model: str) -> float:
    """Estimated USD for one ``analyze`` call (0.0 for local models)."""
    profile = PROFILES.get(profile_key)
    if profile is not None and profile.local:
        return 0.0
    tin, tout = estimate_tokens(req, profile_key, model)
    p_in, p_out = _tiered(model, tin, price_for(model) or UNKNOWN_PRICE)
    return round((tin * p_in + tout * p_out) / 1e6, 6)


def cost_from_usage(model: str, usage: Usage) -> float | None:
    """Actual USD from normalized usage; None when the model's price is unknown."""
    if model == "local" or model.startswith("local:"):
        return 0.0
    price = price_or_conservative(model)
    base = base_model(model)
    total_in = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    p_in, p_out = _tiered(base, total_in, price)
    read_f = CACHE_READ_FACTOR.get(base, 0.10)
    write_f = CACHE_WRITE_FACTOR if base.startswith("claude-") else 1.0
    usd = (
        usage.input_tokens * p_in
        + usage.cache_read_tokens * p_in * read_f
        + usage.cache_write_tokens * p_in * write_f
        + usage.output_tokens * p_out
    ) / 1e6
    return round(usd, 6)


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cache_read_tokens + b.cache_read_tokens,
        a.cache_write_tokens + b.cache_write_tokens,
    )


def sum_costs(costs: Iterable[float | None]) -> float | None:
    vals = [c for c in costs if c is not None]
    return round(sum(vals), 6) if vals else None


class CostCapExceeded(Exception):
    """Raised before a call that would push the run or session past its cap (HTTP 402 in the API)."""


class CostGuard:
    """Per-run and per-session spend caps. Keep one guard per session; call ``new_run()`` for each run."""

    def __init__(
        self, run_cap: float | None = None, session_cap: float | None = None, session_spent: float = 0.0
    ):
        s = get_settings()
        self.run_cap = s.run_cost_cap_usd if run_cap is None else run_cap
        self.session_cap = s.session_cost_cap_usd if session_cap is None else session_cap
        self._run = 0.0
        self._session = session_spent
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        """USD spent in the current run."""
        return self._run

    @property
    def session_spent(self) -> float:
        return self._session

    @property
    def run_remaining(self) -> float:
        return max(0.0, self.run_cap - self._run)

    @property
    def session_remaining(self) -> float:
        return max(0.0, self.session_cap - self._session)

    def check(self, amount: float) -> None:
        """Raise CostCapExceeded if spending ``amount`` more would exceed either cap."""
        amount = max(0.0, amount or 0.0)
        with self._lock:
            if self._run + amount > self.run_cap + 1e-9:
                raise CostCapExceeded(
                    f"This would cost about ${amount:.3f}; the run has ${self.run_remaining:.3f} left "
                    f"of its ${self.run_cap:.2f} cap."
                )
            if self._session + amount > self.session_cap + 1e-9:
                raise CostCapExceeded(
                    f"This would cost about ${amount:.3f}; the session has ${self.session_remaining:.3f} left "
                    f"of its ${self.session_cap:.2f} cap."
                )

    def add(self, amount: float | None) -> None:
        if not amount:
            return
        with self._lock:
            self._run += amount
            self._session += amount

    def new_run(self, run_cap: float | None = None) -> None:
        with self._lock:
            self._run = 0.0
            if run_cap is not None:
                self.run_cap = run_cap


class SessionLedger:
    """Session-wide spend with reservations, so queued runs can't jointly overshoot the cap (thread-safe)."""

    def __init__(self, cap: float | None = None):
        self.cap = get_settings().session_cost_cap_usd if cap is None else cap
        self.spent = 0.0
        self.reserved = 0.0
        self._lock = threading.Lock()

    def reserve(self, amount: float) -> None:
        """Hold ``amount`` for a submitted run; raises CostCapExceeded when spent + held + amount > cap."""
        amount = max(0.0, amount)
        with self._lock:
            if self.spent + self.reserved + amount > self.cap + 1e-9:
                left = max(0.0, self.cap - self.spent - self.reserved)
                raise CostCapExceeded(
                    f"This run is estimated at ${amount:.2f}, but only ${left:.2f} of the ${self.cap:.2f} "
                    "session cap is left (including runs still queued)."
                )
            self.reserved += amount

    def release(self, amount: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - max(0.0, amount))

    def add(self, amount: float | None) -> None:
        if amount:
            with self._lock:
                self.spent += amount

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap - self.spent)

    def run_guard(self, run_cap: float | None = None) -> RunGuard:
        return RunGuard(self, run_cap)


class RunGuard:
    """Per-run cap on top of a SessionLedger: check before each call, add actual cost right after it."""

    def __init__(self, ledger: SessionLedger, run_cap: float | None = None):
        self.ledger = ledger
        self.run_cap = get_settings().run_cost_cap_usd if run_cap is None else run_cap
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self, amount: float) -> None:
        amount = max(0.0, amount or 0.0)
        with self._lock:
            if self.spent + amount > self.run_cap + 1e-9:
                raise CostCapExceeded(
                    f"This would cost about ${amount:.3f}; the run has ${max(0.0, self.run_cap - self.spent):.3f} "
                    f"left of its ${self.run_cap:.2f} cap."
                )
        if self.ledger.spent + amount > self.ledger.cap + 1e-9:
            raise CostCapExceeded(
                f"This would cost about ${amount:.3f}; the session has ${self.ledger.remaining():.3f} left "
                f"of its ${self.ledger.cap:.2f} cap."
            )

    def add(self, amount: float | None) -> None:
        if not amount:
            return
        with self._lock:
            self.spent += amount
        self.ledger.add(amount)
