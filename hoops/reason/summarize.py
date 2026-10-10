"""System Two tasks over video: clip summaries (presets / custom) and adjudication of escalated shot windows."""

from __future__ import annotations

from pathlib import Path

from hoops.reason.base import AnalyzeRequest, AnalyzeResult, Effort, ImagePart, VideoPart, VisionLLM
from hoops.reason.prompts import load_prompt, system_prompt
from hoops.reason.sheets import TILE_RE, mmss, overview_images, resolve_ref, window_images
from hoops.schemas import ClipSummary, WindowVerdict

STRUCTURED_FORMAT = (
    "Fill the fields: headline (one line, under 12 words); narrative (the main text, plain language); "
    "key_moments (ref = the tile id or MM:SS where it happens, description = one sentence); "
    "caveats (what was unclear or where the frames and the CV context disagree; empty if none)."
)
TEXT_FORMAT = (
    "Write Markdown: a bold one-line headline, then the main text, then a short 'Key moments' list where each "
    "item starts with its tile id or MM:SS, then any caveats."
)
VIDEO_NOTE = "You are watching the clip itself (with audio); cite moments as MM:SS."


def summary_prompt(preset: str, custom_prompt: str | None, *, structured: bool, native_video: bool) -> str:
    if preset == "custom" or (custom_prompt and preset not in ("recap", "play_by_play", "coach")):
        task = load_prompt("custom", instructions=custom_prompt or "Summarize this basketball clip.")
    else:
        task = load_prompt(preset)
        if custom_prompt:
            task += f"\n\nAlso: {custom_prompt}"
    parts = [task, STRUCTURED_FORMAT if structured else TEXT_FORMAT]
    if native_video:
        parts.append(VIDEO_NOTE)
    return "\n\n".join(parts)


def render_summary(summary: ClipSummary) -> str:
    """Markdown for people (refs already normalized to 'S2#4 @ 00:12.40' when resolvable)."""
    lines = [f"**{summary.headline.strip()}**", "", summary.narrative.strip()]
    if summary.key_moments:
        lines += ["", "Key moments:"] + [f"- {m.ref}: {m.description}" for m in summary.key_moments]
    if summary.caveats:
        lines += ["", "Caveats:"] + [f"- {c}" for c in summary.caveats]
    return "\n".join(lines).strip()


def normalize_refs(summary: ClipSummary, index: dict[str, float]) -> ClipSummary:
    """Append the resolved clip time to tile refs ('S2#4' -> 'S2#4 @ 00:12.40') so times survive without the index."""
    moments = []
    known = {k.upper() for k in index}
    for m in summary.key_moments:
        is_tile = any(t.group(1).upper() in known for t in TILE_RE.finditer(m.ref))
        t = resolve_ref(m.ref, index) if is_tile else None
        ref = m.ref if t is None or mmss(t) in m.ref else f"{m.ref} @ {mmss(t)}"
        moments.append(m.model_copy(update={"ref": ref}))
    return summary.model_copy(update={"key_moments": moments})


def summarize(
    vlm: VisionLLM,
    *,
    video: Path,
    duration_s: float,
    preset: str = "recap",
    custom_prompt: str | None = None,
    structured: bool = True,
    cv_context: str | None = None,
    native_video: bool = False,
    fps: float = 1.0,
    model: str | None = None,
    effort: Effort = "medium",
    cache_key: str | None = None,
) -> tuple[ClipSummary | None, str, AnalyzeResult]:
    """Summarize a clip. Native video when asked and supported; otherwise 3x3 contact sheets at ``fps``.

    Returns (parsed summary or None, display text, raw AnalyzeResult).
    """
    use_video = native_video and vlm.caps.native_video
    images: list[ImagePart] = []
    index: dict[str, float] = {}
    if not use_video:
        max_sheets = max(1, min(6, vlm.caps.max_images))
        images, index = overview_images(video, duration_s, fps=fps, max_sheets=max_sheets)
    req = AnalyzeRequest(
        prompt=summary_prompt(preset, custom_prompt, structured=structured, native_video=use_video),
        system=system_prompt(),
        images=images,
        video=VideoPart(path=Path(video), fps=fps) if use_video else None,
        cv_context=cv_context,
        output_model=ClipSummary if structured else None,
        effort=effort,
        max_tokens=16000,
        purpose="summary",
        model=model,
        cache_key=cache_key,
    )
    res = vlm.analyze(req)
    summary = res.parsed if isinstance(res.parsed, ClipSummary) else None
    if summary is not None:
        summary = normalize_refs(summary, index)
        res.parsed = summary
        return summary, render_summary(summary), res
    return None, res.text, res


def adjudicate(
    vlm: VisionLLM,
    *,
    video: Path,
    t0: float,
    t1: float,
    cv_context: str,
    rim_box: tuple | None = None,
    model: str | None = None,
    cache_key: str | None = None,
) -> tuple[WindowVerdict | None, AnalyzeResult]:
    """Ask System Two to settle one escalated shot window [t0, t1].

    Native-video providers get the trimmed window at 8 fps. Others get rim-crop sheets ("R1", ...) at 8 fps when
    ``rim_box`` is known (in ``video``'s pixels: pass the proxy with proxy-pixel boxes) plus one wide sheet ("W1").
    Confidence is clamped to [0, 1] in code.
    """
    t0 = max(0.0, t0)
    images: list[ImagePart] = []
    video_part = None
    if vlm.caps.native_video:
        video_part = VideoPart(path=Path(video), t0_s=t0, t1_s=t1, fps=8.0)
    else:
        if rim_box is not None:
            crops, _ = window_images(video, t0, t1, fps=8.0, sheet_prefix="R", crop_box=rim_box)
            images += crops
        span = max(t1 - t0, 0.1)
        wide, _ = window_images(video, t0, t1, fps=4 / span, sheet_prefix="W")
        images += wide[:1]
        images = images[: vlm.caps.max_images]
    req = AnalyzeRequest(
        prompt=load_prompt("adjudicate", t0=mmss(t0), t1=mmss(t1)),
        system=system_prompt(),
        images=images,
        video=video_part,
        cv_context=cv_context,
        output_model=WindowVerdict,
        effort="low",
        max_tokens=8000,
        purpose="adjudicate",
        model=model,
        cache_key=cache_key,
    )
    res = vlm.analyze(req)
    verdict = res.parsed if isinstance(res.parsed, WindowVerdict) else None
    if verdict is not None:
        verdict = verdict.model_copy(update={"confidence": min(1.0, max(0.0, verdict.confidence))})
        res.parsed = verdict
    return verdict, res
