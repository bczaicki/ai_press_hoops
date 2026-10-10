"""`hoops doctor`: is this Mac ready? Checks tools, filters, GPU, keys (names only), and local services.

Every check is isolated: one failing probe produces a failed Check, never an exception.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from hoops.config import ROOT, get_settings

Severity = Literal["error", "warn", "info"]

FILTERS: dict[str, tuple[Severity, str]] = {
    "zscale": ("warn", "HDR phone video can't be tone-mapped well (fallbacks: scale_vt, or none)"),
    "tonemap": ("warn", "HDR phone video can't be tone-mapped well"),
    "bwdif": ("warn", "interlaced broadcast captures can't be deinterlaced"),
    "idet": ("warn", "interlacing can't be auto-detected"),
    "drawtext": ("info", "ffmpeg can't burn text (Pillow is used instead)"),
    "scale_vt": ("info", "no VideoToolbox HDR fallback"),
}
ENCODERS: dict[str, tuple[Severity, str]] = {
    "libx264": ("error", "can't write browser-playable H.264"),
    "h264_videotoolbox": ("info", "no hardware H.264 encoder for long games"),
}
KEYS: dict[str, str] = {
    "anthropic_api_key": "Claude",
    "openai_api_key": "OpenAI",
    "gemini_api_key": "Gemini",
    "openrouter_api_key": "OpenRouter",
    "typesafe_api_key": "TypeSafe Jev (System One)",
    "roboflow_api_key": "Roboflow basketball weights",
}
FFMPEG_FIX = (
    "brew install ffmpeg-full (keg-only: set HOOPS_FFMPEG to its bin/ffmpeg), or a 7.1 build with libzimg"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None
    severity: Severity = "error"


def _ffmpeg_checks() -> list[Check]:
    from hoops.ingest import ffmpeg_capabilities

    s = get_settings()
    out: list[Check] = []
    probe = shutil.which(s.ffprobe_bin)
    out.append(Check("ffprobe", bool(probe), probe or "not found", None if probe else "brew install ffmpeg"))
    if not shutil.which(s.ffmpeg_bin):
        out.append(Check("ffmpeg", False, f"'{s.ffmpeg_bin}' not found", "brew install ffmpeg"))
        return out
    caps = ffmpeg_capabilities(s.ffmpeg_bin)
    out.append(Check("ffmpeg", True, f"version {caps.version}"))
    for name, (sev, missing) in FILTERS.items():
        ok = name in caps.filters
        out.append(
            Check(f"filter:{name}", ok, "available" if ok else missing, None if ok else FFMPEG_FIX, sev)
        )
    for name, (sev, missing) in ENCODERS.items():
        ok = name in caps.encoders
        out.append(
            Check(f"encoder:{name}", ok, "available" if ok else missing, None if ok else FFMPEG_FIX, sev)
        )
    return out


def _brew_pinned() -> Check:
    if not shutil.which("brew"):
        return Check("ffmpeg pinned", True, "Homebrew not installed; skipped", severity="info")
    pinned = subprocess.run(
        ["brew", "list", "--pinned"], check=False, capture_output=True, text=True, timeout=30
    ).stdout.split()
    ok = any(p in pinned for p in ("ffmpeg", "ffmpeg-full"))
    return Check(
        "ffmpeg pinned",
        ok,
        "pinned in Homebrew" if ok else "not pinned: `brew upgrade` may swap in a build without zscale",
        None if ok else "brew pin ffmpeg",
        "warn",
    )


def _torch_mps() -> Check:
    if importlib.util.find_spec("torch") is None:
        return Check("torch MPS", False, "torch not installed", "uv sync --extra cv", "warn")
    import torch

    ok = bool(torch.backends.mps.is_available())
    detail = f"torch {torch.__version__}: " + ("Apple GPU available" if ok else "MPS unavailable, using CPU")
    return Check("torch MPS", ok, detail, None if ok else "update macOS/torch; CPU still works", "warn")


def _mps_fallback() -> Check:
    ok = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    return Check(
        "PYTORCH_ENABLE_MPS_FALLBACK",
        ok,
        "set" if ok else "not set: an unsupported GPU op will crash instead of falling back to CPU",
        None if ok else "export PYTORCH_ENABLE_MPS_FALLBACK=1 (or add it to .env)",
        "warn",
    )


def _key_checks() -> list[Check]:
    s = get_settings()
    out = []
    for field, label in KEYS.items():
        ok = s.key(field) is not None
        detail = f"{label}: set" if ok else f"{label}: missing, so it's disabled"
        out.append(
            Check(f"key:{field.upper()}", ok, detail, None if ok else f"add {field.upper()} to .env", "warn")
        )
    pw = s.app_password is not None and bool(s.app_password.get_secret_value())
    out.append(
        Check(
            "HOOPS_APP_PASSWORD",
            pw,
            "set" if pw else "not set: anyone with a share link could use the app (and your API credit)",
            None if pw else "add HOOPS_APP_PASSWORD to .env before sharing",
            "warn",
        )
    )
    return out


def _ollama() -> Check:
    import httpx

    host = get_settings().ollama_host.rstrip("/")
    try:
        r = httpx.get(f"{host}/api/tags", timeout=1.5)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return Check(
            "ollama", False, f"not reachable at {host} ({type(exc).__name__})", "open the Ollama app", "warn"
        )
    has_qwen = any(m.startswith("qwen3.5") for m in models)
    return Check(
        "ollama",
        has_qwen,
        f"reachable; {len(models)} model(s)" + ("" if has_qwen else ", no qwen3.5"),
        None if has_qwen else "ollama pull qwen3.5:4b",
        "warn",
    )


def _web_build() -> Check:
    ok = (ROOT / "web" / "dist" / "index.html").exists()
    return Check(
        "web build", ok, "web/dist/index.html found" if ok else "web UI not built", None if ok else
        "cd web && npm install && npm run build", "warn",
    )  # fmt: skip


def _data_dir() -> Check:
    d = get_settings().data_dir
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".doctor_write_test"
        probe.write_text("ok")
        probe.unlink()
        free_gb = shutil.disk_usage(d).free / 1e9
    except OSError as exc:
        return Check("data dir", False, f"{d} is not writable ({exc.strerror})", "set HOOPS_DATA_DIR")
    if free_gb < 5:
        return Check("data dir", False, f"{d}: only {free_gb:.1f} GB free", "free some disk space", "warn")
    return Check("data dir", True, f"{d} ({free_gb:.0f} GB free)")


def _plugins() -> list[Check]:
    from hoops.registry import load_plugins

    errors = dict(load_plugins())
    if not errors:
        return [Check("plugins", True, "all implementation modules import", severity="warn")]
    return [Check(f"plugin:{mod}", False, err, "see the error; usually a missing uv extra", "warn")
            for mod, err in sorted(errors.items())]  # fmt: skip


def run_doctor() -> list[Check]:
    checks = [
        Check("python", sys.version_info >= (3, 12), f"{platform.python_version()} on {platform.platform()}")
    ]
    steps: list[tuple[str, Callable[[], Check | list[Check]]]] = [
        ("ffmpeg", _ffmpeg_checks),
        ("ffmpeg pinned", _brew_pinned),
        ("torch MPS", _torch_mps),
        ("PYTORCH_ENABLE_MPS_FALLBACK", _mps_fallback),
        ("keys", _key_checks),
        ("ollama", _ollama),
        ("web build", _web_build),
        ("data dir", _data_dir),
        ("plugins", _plugins),
    ]
    for name, fn in steps:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - doctor must always finish
            result = Check(name, False, f"check crashed: {type(exc).__name__}: {exc}", severity="warn")
        checks += result if isinstance(result, list) else [result]
    return checks


def render_text(checks: list[Check]) -> str:
    """Plain-text report for the CLI: one line per check, fixes indented underneath."""
    icon = {"error": "FAIL", "warn": "WARN", "info": "info"}
    lines = []
    for c in checks:
        lines.append(f"[{' OK ' if c.ok else icon[c.severity]}] {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"       fix: {c.fix}")
    errors = sum(1 for c in checks if not c.ok and c.severity == "error")
    warns = sum(1 for c in checks if not c.ok and c.severity == "warn")
    lines.append(f"\n{errors} error(s), {warns} warning(s)")
    return "\n".join(lines)
