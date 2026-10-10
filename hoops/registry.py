"""String-keyed registries for swappable implementations (detectors, trackers, deciders, VLMs).

Implementation modules register factories at import time and keep heavy imports (torch, rfdetr, SDKs)
inside the factory, so importing a module is cheap and never fails because an optional extra is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hoops.config import get_settings


@dataclass
class Entry:
    key: str
    label: str
    factory: Callable[..., Any]
    license: str = ""
    requires_modules: tuple[str, ...] = ()
    requires_keys: tuple[str, ...] = ()  # Settings field names, e.g. "anthropic_api_key"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def unavailable_reason(self) -> str | None:
        for mod in self.requires_modules:
            if importlib.util.find_spec(mod) is None:
                return f"missing Python package '{mod}' (install the matching uv extra)"
        settings = get_settings()
        for k in self.requires_keys:
            if not settings.key(k):
                return f"missing {k.upper()} in .env"
        return None


class Registry[T]:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry] = {}

    def register(
        self,
        key: str,
        *,
        label: str,
        license: str = "",
        requires_modules: tuple[str, ...] = (),
        requires_keys: tuple[str, ...] = (),
        notes: str = "",
        **extra: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            self._entries[key] = Entry(
                key, label, factory, license, requires_modules, requires_keys, notes, extra
            )
            return factory

        return deco

    def create(self, key: str, **kwargs: Any) -> T:
        if key not in self._entries:
            raise KeyError(f"unknown {self.kind} '{key}'. Known: {sorted(self._entries)}")
        entry = self._entries[key]
        reason = entry.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.kind} '{key}' unavailable: {reason}")
        return entry.factory(**kwargs)

    def entry(self, key: str) -> Entry:
        return self._entries[key]

    def entries(self) -> list[Entry]:
        return list(self._entries.values())

    def available(self) -> dict[str, str | None]:
        """key -> None if usable, else the reason it isn't."""
        return {k: e.unavailable_reason() for k, e in self._entries.items()}


detectors: Registry[Any] = Registry("detector")
trackers: Registry[Any] = Registry("tracker")
deciders: Registry[Any] = Registry("decider")
vlms: Registry[Any] = Registry("vlm")

# Modules that register implementations. Import failures are recorded, not raised.
PLUGIN_MODULES = (
    "hoops.perception.detectors",
    "hoops.perception.tracking",
    "hoops.decide.rules",
    "hoops.decide.adapter",
    "hoops.decide.jev",
    "hoops.decide.openai_decisions",
    "hoops.reason.anthropic_vlm",
    "hoops.reason.gemini_vlm",
    "hoops.reason.openai_compat",
)
PLUGIN_ERRORS: dict[str, str] = {}


def load_plugins() -> dict[str, str]:
    for mod in PLUGIN_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - surface in `hoops doctor`, keep the app running
            PLUGIN_ERRORS[mod] = f"{type(exc).__name__}: {exc}"
    return PLUGIN_ERRORS
