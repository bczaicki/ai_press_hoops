"""Helpers shared by System Two adapters: prompt assembly, JSON recovery, image fitting, clip trimming."""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError

from hoops.config import get_settings
from hoops.reason.base import AnalyzeRequest, AnalyzeResult, ImagePart, Usage

CV_HEADER = "CV context (from detection and tracking; ground truth for counts, scores, positions and times):"
EFFORT_3 = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def now() -> float:
    return time.perf_counter()


def elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def user_text(req: AnalyzeRequest, *, schema_hint: bool = False) -> str:
    """CV context first, then the task (long context before the question), then an optional schema hint."""
    parts = []
    if req.cv_context:
        parts.append(f"{CV_HEADER}\n{req.cv_context}")
    parts.append(req.prompt)
    if schema_hint and req.output_model is not None:
        parts.append(schema_instructions(req.output_model))
    return "\n\n".join(p for p in parts if p)


def schema_instructions(model: type[BaseModel]) -> str:
    schema = json.dumps(inline_refs(model.model_json_schema()), separators=(",", ":"))
    return (
        "Reply with only one JSON object (no prose, no code fences) that validates against this JSON Schema:\n"
        f"{schema}"
    )


def inline_refs(schema: dict) -> dict:
    """Resolve local ``$ref``s into a self-contained schema (some providers reject ``$defs``)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def extract_json(text: str) -> str:
    """Strip code fences / surrounding prose from a model reply that should be one JSON object."""
    t = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fenced:
        t = fenced.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            t = t[i : j + 1]
    return t


def parse_output(text: str, model: type[BaseModel]) -> tuple[BaseModel | None, str | None]:
    """(parsed, None) or (None, error message)."""
    try:
        return model.model_validate_json(extract_json(text)), None
    except (ValidationError, ValueError) as exc:
        msg = str(exc).splitlines()
        return None, f"reply did not match {model.__name__}: {' '.join(msg[:3])[:400]}"


def fit_jpeg(jpeg: bytes, max_px: int) -> bytes:
    """Downscale a JPEG so its long edge is at most ``max_px`` (no-op when it already fits)."""
    with Image.open(io.BytesIO(jpeg)) as img:
        if max(img.size) <= max_px:
            return jpeg
        img = img.convert("RGB")
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def limit_images(images: list[ImagePart], n: int) -> list[ImagePart]:
    """Keep at most ``n`` images, evenly spaced (keeps the first and last)."""
    if len(images) <= n:
        return list(images)
    if n <= 1:
        return images[:1]
    step = (len(images) - 1) / (n - 1)
    return [images[round(i * step)] for i in range(n)]


def history_text(history: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"Earlier question: {q}\nYour earlier answer: {a}" for q, a in history)


def to_jsonable(obj: Any) -> Any:
    """Best-effort JSON-serializable view of an SDK response object."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                data = fn(mode="json") if attr == "model_dump" else fn()
                return json.loads(json.dumps(data, default=str))
            except Exception:  # noqa: BLE001, S112 - try the next serializer
                continue
    return str(obj)


def failure(
    provider: str,
    model: str,
    t0: float,
    error: str,
    *,
    usage: Usage | None = None,
    cost: float | None = None,
    raw: Any = None,
    text: str = "",
    stop_reason: str | None = None,
) -> AnalyzeResult:
    return AnalyzeResult(
        text=text,
        parsed=None,
        usage=usage or Usage(),
        cost_usd=cost,
        latency_ms=elapsed_ms(t0),
        model_id=model,
        provider=provider,
        stop_reason=stop_reason,
        error=error,
        raw=raw,
    )


def short_error(exc: BaseException) -> str:
    msg = getattr(exc, "message", None) or str(exc)
    return f"{type(exc).__name__}: {msg}"[:500]


def trim_clip(src: Path, t0: float, t1: float, out: Path | None = None) -> Path:
    """Re-encode [t0, t1) of ``src`` into a small H.264/AAC mp4 (accurate cut; no provider offsets needed)."""
    s = get_settings()
    if out is None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="hoops_clip_", delete=False) as f:
            out = Path(f.name)
    cmd = [
        s.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-protocol_whitelist", "file",
        "-ss", f"{max(0.0, t0):.3f}", "-i", str(src), "-t", f"{max(0.1, t1 - max(0.0, t0)):.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    return out
