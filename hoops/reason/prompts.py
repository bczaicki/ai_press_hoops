"""Prompt templates in ``hoops/reason/prompts/*.md``.

Each file may start with a ``---`` header of ``key: value`` lines (label, preset, description). Templates use
``{{name}}`` placeholders (so JSON braces in prompts are safe); unknown or None values render as "".
``{{rules}}`` expands to ``_rules.md``. Files are read on every call so edits from the UI apply immediately.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).with_name("prompts")
PRESET_ORDER = ("recap", "play_by_play", "coach", "custom")
_NAME = re.compile(r"[a-z_][a-z0-9_]*")
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _read(name: str) -> tuple[dict[str, str], str]:
    path = PROMPT_DIR / f"{name}.md"
    if not _NAME.fullmatch(name) or not path.is_file():
        raise KeyError(f"unknown prompt '{name}'")
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    if text.startswith("---\n"):
        head, _, text = text[4:].partition("\n---\n")
        for line in head.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                meta[key.strip()] = value.strip()
    return meta, text.strip()


def load_prompt(name: str, **vars: Any) -> str:
    """Render prompt ``name`` with ``vars``."""
    _, body = _read(name)

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key == "rules" and "rules" not in vars:
            return _read("_rules")[1]
        value = vars.get(key)
        return "" if value is None else str(value)

    return re.sub(r"\n{3,}", "\n\n", _VAR.sub(sub, body)).strip()


def prompt_meta(name: str) -> dict[str, str]:
    return _read(name)[0]


def list_presets() -> list[dict]:
    """User-facing summary presets, e.g. [{"key": "recap", "label": "Recap", "description": ...}, ...]."""
    out = []
    for path in PROMPT_DIR.glob("*.md"):
        meta, _ = _read(path.stem)
        if meta.get("preset", "").lower() == "true":
            out.append(
                {
                    "key": path.stem,
                    "label": meta.get("label", path.stem),
                    "description": meta.get("description", ""),
                }
            )
    rank = {k: i for i, k in enumerate(PRESET_ORDER)}
    return sorted(out, key=lambda p: (rank.get(p["key"], len(rank)), p["key"]))


def system_prompt() -> str:
    return load_prompt("system")
