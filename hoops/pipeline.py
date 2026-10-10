"""Orchestrates one analysis run across the tiers.

T1 perception -> shots -> System One decisions -> (escalated) System Two adjudication -> stats ->
summary -> court map -> annotated video. Each stage degrades gracefully: a failure becomes a run
warning and a failed stage, and the run continues with what it has.

Stage modules are imported lazily so the server starts even when an optional extra is missing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from hoops import registry
from hoops.config import PROFILES, get_settings
from hoops.decide.base import PRESETS
from hoops.jobs import Cancelled, JobContext
from hoops.schemas import (
    AnalysisRun,
    ClipMeta,
    FootageType,
    Narrative,
    OutputKind,
    Provenance,
    StageConfig,
    VLMCallRecord,
)
from hoops.store import Store

log = logging.getLogger(__name__)


class VLMChoice(BaseModel):
    provider: str
    model: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    native_video: bool = False


class CourtPoint(BaseModel):
    name: str
    x: float
    y: float


class RunRequest(BaseModel):
    clip_id: str
    outputs: list[OutputKind] = Field(default_factory=lambda: ["annotated_video", "events", "summary"])
    detector: str = "rfdetr:coco"
    tracker: str = "bytetrack"
    decider: str = "rules"
    vlm: VLMChoice | None = None
    prompt_preset: str = "recap"
    custom_prompt: str | None = None
    structured: bool = True
    threshold_preset: str = "balanced"
    egress_ok: list[str] = Field(default_factory=list)
    footage_type: FootageType | None = None
    rim_box: list[float] | None = None
    court_points: list[CourtPoint] | None = None
    max_cost_usd: float | None = None
    stride: int = 1
    max_escalations: int = 12


class EstimateLine(BaseModel):
    stage: str
    provider: str
    model: str
    usd: float
    note: str = ""


class Estimate(BaseModel):
    estimate_usd: float
    breakdown: list[EstimateLine]
    warnings: list[str] = Field(default_factory=list)


def stages_for(req: RunRequest) -> list[str]:
    needs_cv = bool(set(req.outputs) & {"annotated_video", "events", "court_map"})
    stages = []
    if needs_cv:
        stages += ["detect", "track", "shots", "decide"]
        if req.vlm:
            stages.append("adjudicate")
    if "summary" in req.outputs:
        stages.append("summarize")  # skipped (in place) when no AI provider was chosen
    if "court_map" in req.outputs:
        stages.append("court")
    if "annotated_video" in req.outputs:
        stages.append("render")
    return stages


# ---------------------------------------------------------------------------------------- estimate


def estimate(req: RunRequest, clip: ClipMeta) -> Estimate:
    from hoops.reason.base import AnalyzeRequest, ImagePart

    lines: list[EstimateLine] = []
    warnings: list[str] = []
    duration = clip.master.duration_s
    if req.vlm:
        profile = PROFILES.get(req.vlm.provider)
        if profile is None:
            warnings.append(f"Unknown AI provider '{req.vlm.provider}'")
        else:
            model = req.vlm.model or default_model(profile.key)
            if req.vlm.provider not in req.egress_ok and not profile.local:
                warnings.append(f"Allow sending frames to {profile.sends_data_to} to use {profile.label}")
            try:
                from hoops import cost

                dummy = ImagePart(jpeg=b"", label="")
                if "summary" in req.outputs:
                    if req.vlm.native_video and profile.native_video:
                        from hoops.reason.base import VideoPart

                        areq = AnalyzeRequest(
                            prompt="x" * 2000, video=VideoPart(path=Path("x"), fps=1.0), max_tokens=4000
                        )
                        usd = cost.estimate(areq, profile.key, model) * max(duration / 60.0, 0.1)
                    else:
                        n = min(max(int(duration), 4), 6 * 9)  # ~1 fps overview packed into <=6 sheets
                        sheets = max(1, -(-n // 9))
                        areq = AnalyzeRequest(prompt="x" * 3000, images=[dummy] * sheets, max_tokens=4000)
                        usd = cost.estimate(areq, profile.key, model)
                    lines.append(
                        EstimateLine(
                            stage="summarize",
                            provider=profile.key,
                            model=model,
                            usd=usd,
                            note="recap of the whole clip",
                        )
                    )
                if set(req.outputs) & {"events", "court_map", "annotated_video"}:
                    expected_shots = max(1, int(duration / 12))
                    escalations = min(req.max_escalations, max(1, round(expected_shots * 0.4)))
                    areq = AnalyzeRequest(
                        prompt="x" * 1500, images=[dummy] * 4, max_tokens=1500, effort="low"
                    )
                    usd = cost.estimate(areq, profile.key, model) * escalations
                    lines.append(
                        EstimateLine(
                            stage="adjudicate",
                            provider=profile.key,
                            model=model,
                            usd=usd,
                            note=f"~{escalations} close calls reviewed",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Cost estimate unavailable: {exc}")
    if req.decider.startswith("adapter:"):
        expected = max(1, int(duration / 12))
        lines.append(
            EstimateLine(
                stage="decide",
                provider=req.decider,
                model="llm",
                usd=0.004 * expected,
                note="LLM-backed System One (Jev is ~free)",
            )
        )
    total = round(sum(line.usd for line in lines), 4)
    cap = min(req.max_cost_usd or get_settings().run_cost_cap_usd, get_settings().run_cost_cap_usd)
    if total > cap:
        warnings.append(f"Estimated ${total:.2f} exceeds the ${cap:.2f} per-run cap")
    return Estimate(estimate_usd=total, breakdown=lines, warnings=warnings)


# ---------------------------------------------------------------------------------------- run


def run_analysis(
    req: RunRequest, ctx: JobContext | None, store: Store, run_id: str | None = None, guard: Any = None
) -> AnalysisRun:
    """Execute a run end to end. ``ctx`` may be None (CLI use). ``guard`` is a hoops.cost.RunGuard
    (the server passes one tied to the session ledger); without it, a fresh ledger caps this run alone."""
    t_start = time.monotonic()
    clip = store.load_clip(req.clip_id)
    if clip is None:
        raise ValueError(f"Unknown clip {req.clip_id}")
    if req.vlm and req.vlm.provider not in req.egress_ok and not PROFILES[req.vlm.provider].local:
        raise PermissionError(
            f"Frames can't be sent to {PROFILES[req.vlm.provider].sends_data_to} without opt-in"
        )

    run = AnalysisRun(run_id=run_id or store.new_run_id(), clip=clip, outputs_requested=list(req.outputs))
    if req.footage_type:
        run.clip.footage_type, run.clip.footage_type_source = req.footage_type, "human"
    run_dir = store.run_dir(run.run_id)
    (run_dir / "request.json").write_text(req.model_dump_json(indent=2))
    store.save_run(run)
    p = _Progress(ctx)

    proxy = store.abs(clip.proxy.path)
    master = store.abs(clip.master.path)
    preset = PRESETS.get(req.threshold_preset, PRESETS["balanced"])
    if guard is None:
        from hoops.cost import SessionLedger

        guard = SessionLedger().run_guard(_run_cap(req))
    per_call = _per_call_estimates(req, clip)
    vlm = _make_vlm(req, run) if req.vlm else None
    decider = _make_decider(req.decider, run)

    detections = None
    candidates: list[Any] = []
    escalate: list[Any] = []
    rim_box = tuple(req.rim_box) if req.rim_box else None
    needs_cv = bool(set(req.outputs) & {"annotated_video", "events", "court_map"})

    # ---- T1 perception
    if needs_cv:
        p.start("detect")
        try:
            from hoops.perception.run import PerceptionCancelled, run_perception

            try:
                perc = run_perception(
                    proxy,
                    detector_key=req.detector,
                    tracker_key=req.tracker,
                    stride=max(1, req.stride),
                    scene_cuts_s=clip.scene_cuts_s,
                    rim_box=rim_box,
                    progress=p.cb("detect"),
                    cancel=(lambda: ctx.cancelled) if ctx else None,
                )
            except PerceptionCancelled as exc:
                raise Cancelled() from exc
            detections = perc.detections
            run.tracks = perc.tracks
            rim_box = perc.rim_box or rim_box
            run.warnings += perc.warnings
            run.metrics.detect_fps = perc.fps_measured
            det_path = run_dir / "detections.parquet"
            detections.to_parquet(det_path, index=False)
            run.detections_path = store.rel(det_path)
            dinfo = getattr(perc, "detector_info", None) or {}
            tinfo = getattr(perc, "tracker_info", None) or {}
            run.pipeline.append(
                StageConfig(
                    stage="detect",
                    impl=req.detector,
                    version=_s(dinfo.get("version")),
                    license=_s(dinfo.get("license")),
                    params={"stride": req.stride, "device": _s(dinfo.get("device"))},
                )
            )
            run.pipeline.append(
                StageConfig(
                    stage="track",
                    impl=req.tracker,
                    version=_s(tinfo.get("version")),
                    license=_s(tinfo.get("license")),
                )
            )
            p.done("detect", f"{perc.frames_processed} frames at {perc.fps_measured:.1f} fps")
            p.done("track")
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            _stage_failed(run, p, "detect", exc)
            p.skip("track")

        # Teams (best effort, part of tracking)
        if detections is not None and len(detections):
            try:
                from hoops.perception.teams import assign_teams, track_teams

                detections, run.teams = assign_teams(proxy, detections, method="hsv")
                detections.to_parquet(run_dir / "detections.parquet", index=False)
                team_of = track_teams(detections)
                for tr in run.tracks:
                    tr.team_id = team_of.get(tr.track_id) or tr.team_id
            except Exception as exc:  # noqa: BLE001
                run.warnings.append(f"Team split skipped: {exc}")

        # Footage triage (System One) when the user didn't set it
        if run.clip.footage_type == "unknown" and detections is not None:
            _triage(run, decider, detections, clip)

        # No rim marked and none detected (e.g. COCO): suggest hoops zero-shot, keep the one the ball approaches
        if detections is not None and rim_box is None and not (detections["cls"] == "rim").any():
            rim_box = _auto_rim(run, proxy, detections)

        # Shots
        if detections is not None:
            p.start("shots")
            try:
                from hoops.events.shots import detect_candidates

                candidates = detect_candidates(detections, rim_box=rim_box, scene_cuts_s=clip.scene_cuts_s)
                if rim_box is None and not (detections["cls"] == "rim").any():
                    run.warnings.append("No rim found: mark the rim on the clip to detect shots")
                p.done("shots", f"{len(candidates)} candidate shots")
            except Exception as exc:  # noqa: BLE001
                _stage_failed(run, p, "shots", exc)

        # System One
        if candidates:
            p.start("decide")
            try:
                from hoops.events.escalate import type_candidates

                images_for = None
                decider_provider = DECIDER_PROVIDERS.get(req.decider)
                if getattr(decider, "supports_images", False):
                    if decider_provider in req.egress_ok:
                        images_for = _rim_crops_fn(proxy)
                        run.data_egress = sorted(set(run.data_egress) | {decider_provider})
                    else:
                        run.warnings.append(
                            f"System One '{req.decider}' ran on text only: allow {decider_provider} to send it "
                            "rim crops"
                        )
                events, records, escalate = type_candidates(
                    candidates,
                    decider,
                    preset=preset,
                    run_id=run.run_id,
                    max_escalations=req.max_escalations,
                    images_for=images_for,
                )
                run.events, run.decisions = list(events), run.decisions + list(records)
                decide_cost = sum(r.cost_usd or 0.0 for r in records)
                if decide_cost:
                    guard.add(decide_cost)
                    run.metrics.cost_usd = round(run.metrics.cost_usd + decide_cost, 6)
                run.pipeline.append(
                    StageConfig(stage="decide", impl=req.decider, params={"preset": preset.name})
                )
                p.done("decide", f"{len(events)} events, {len(escalate)} close calls")
            except Exception as exc:  # noqa: BLE001
                _stage_failed(run, p, "decide", exc)
        else:
            p.skip("decide", "No shot candidates")

        # System Two adjudication of close calls
        if vlm and escalate:
            p.start("adjudicate")
            try:
                from hoops.events.escalate import apply_verdicts
                from hoops.reason.summarize import adjudicate

                verdicts = {}
                scale = clip.master.width / max(clip.proxy.width, 1)
                for i, cand in enumerate(escalate):
                    p.progress(
                        "adjudicate", i / len(escalate), f"Reviewing close call {i + 1}/{len(escalate)}"
                    )
                    guard.check(per_call["adjudicate"])
                    box = tuple(v * scale for v in cand.rim_box) if getattr(cand, "rim_box", None) else None
                    verdict, result = adjudicate(
                        vlm,
                        video=master,
                        t0=cand.t_start_s,
                        t1=cand.t_end_s,
                        cv_context=json.dumps({"descriptors": getattr(cand, "descriptors", {})}),
                        rim_box=box,
                        model=req.vlm.model,
                        cache_key=run.run_id,
                    )
                    _record_vlm(run, store, result, "adjudicate", guard)
                    if verdict is not None:
                        verdicts[cand.cand_id] = verdict
                run.events = apply_verdicts(
                    run.events, verdicts, producer=f"{req.vlm.provider}:{_model(req)}"
                )
                p.done("adjudicate", f"{len(verdicts)} close calls reviewed")
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                _stage_failed(run, p, "adjudicate", exc)
        elif "adjudicate" in stages_for(req):
            p.skip("adjudicate", "No close calls")

        # Stats (always cheap)
        try:
            from hoops.events.stats import compute_stats

            run.team_summaries = compute_stats(run.events, run.teams)
        except Exception as exc:  # noqa: BLE001
            run.warnings.append(f"Stats unavailable: {exc}")
        store.save_run(run)

    # ---- Summary (System Two)
    if "summary" in req.outputs and vlm:
        p.start("summarize")
        try:
            from hoops.reason.summarize import summarize

            guard.check(per_call["summarize"])
            profile = PROFILES[req.vlm.provider]
            summary, text, result = summarize(
                vlm,
                video=master,
                duration_s=clip.master.duration_s,
                preset=req.prompt_preset,
                custom_prompt=req.custom_prompt,
                structured=req.structured,
                cv_context=_cv_context(run),
                native_video=req.vlm.native_video and profile.native_video,
                model=req.vlm.model,
                effort=req.vlm.effort,
                cache_key=run.run_id,
            )
            _record_vlm(run, store, result, "summary", guard)
            if result.error or result.refused:
                run.warnings.append(f"AI summary: {result.error or 'the model declined this request'}")
            body = text
            if summary is not None:
                body = f"### {summary.headline}\n\n{summary.narrative}"
            if body:
                run.narratives.append(
                    Narrative(
                        kind=NARRATIVE_KINDS.get(req.prompt_preset, "custom"),
                        text=body,
                        provenance=Provenance(
                            source="vlm",
                            producer=f"{req.vlm.provider}:{result.model_id}",
                            stage="summarize",
                            prompt_id=req.prompt_preset,
                            params={
                                "structured": req.structured,
                                "summary": summary.model_dump() if summary else None,
                                "key_moments": moments_with_times(summary) if summary else [],
                            },
                        ),
                    )
                )
            p.done("summarize")
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            _stage_failed(run, p, "summarize", exc)
    elif "summary" in req.outputs:
        p.skip("summarize", "No AI provider selected")
        run.warnings.append("Pick an AI provider to get a summary and Q&A")

    # ---- Court map
    if "court_map" in req.outputs:
        p.start("court")
        try:
            run.court_map_paths = _court_maps(run, req, store, detections, proxy)
            p.done("court")
        except Exception as exc:  # noqa: BLE001
            _stage_failed(run, p, "court", exc)

    # ---- Annotated video + captions
    if "annotated_video" in req.outputs and detections is not None:
        p.start("render")
        try:
            from hoops.render import render_annotated

            out = run_dir / "annotated.mp4"
            colors = {t.team_id: t.color_hex for t in run.teams if t.color_hex}
            render_annotated(
                proxy,
                detections,
                out,
                events=run.events,
                team_colors=colors or None,
                rim_box=rim_box,
                master_audio=master if clip.master.has_audio else None,
                progress=p.cb("render"),
            )
            run.annotated_video_path = store.rel(out)
            p.done("render")
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            _stage_failed(run, p, "render", exc)
    elif "annotated_video" in req.outputs:
        p.skip("render", "Detection failed")
    if run.events:
        try:
            from hoops.captions import write_vtt

            vtt = write_vtt(run.events, run_dir / "events.vtt")
            run.captions_path = store.rel(vtt)
        except Exception as exc:  # noqa: BLE001
            run.warnings.append(f"Captions skipped: {exc}")

    run.metrics.wall_s = round(time.monotonic() - t_start, 2)
    run.status = "done"
    store.save_run(run)
    return run


NARRATIVE_KINDS = {
    "recap": "recap",
    "play_by_play": "play_by_play",
    "coach": "coaching",
    "coaching": "coaching",
}
CLOCK_RE = re.compile(r"(\d+):(\d{2}(?:\.\d+)?)")


def moments_with_times(summary: Any) -> list[dict]:
    """Key moments as dicts with t_s parsed from refs like 'S2#4 @ 00:12.40' or '0:12'."""
    out = []
    for km in getattr(summary, "key_moments", []) or []:
        m = CLOCK_RE.search(km.ref or "")
        out.append(
            {
                "ref": km.ref,
                "description": km.description,
                "t_s": int(m.group(1)) * 60 + float(m.group(2)) if m else None,
            }
        )
    return out


def guard_check(guard: Any, amount: float) -> None:
    guard.check(amount)


# ---------------------------------------------------------------------------------------- helpers


class _Progress:
    """Null-safe wrapper over JobContext."""

    def __init__(self, ctx: JobContext | None):
        self.ctx = ctx

    def start(self, stage: str, msg: str | None = None) -> None:
        if self.ctx:
            self.ctx.check_cancel()
            self.ctx.start(stage, msg)

    def progress(self, stage: str, f: float, msg: str | None = None) -> None:
        if self.ctx:
            self.ctx.check_cancel()
            self.ctx.progress(stage, f, msg)

    def done(self, stage: str, msg: str | None = None) -> None:
        if self.ctx:
            self.ctx.done(stage, msg)

    def skip(self, stage: str, msg: str | None = None) -> None:
        if self.ctx:
            self.ctx.skip(stage, msg)

    def fail(self, stage: str, msg: str) -> None:
        if self.ctx:
            self.ctx.fail(stage, msg)

    def cb(self, stage: str):
        if self.ctx:
            return self.ctx.callback(stage)
        return lambda f, m="": None


def _s(value: object) -> str | None:
    return None if value is None else str(value)


def _stage_failed(run: AnalysisRun, p: _Progress, stage: str, exc: Exception) -> None:
    if isinstance(exc, Cancelled):
        raise exc
    log.exception("stage %s failed", stage)
    run.warnings.append(f"{stage}: {exc}")
    p.fail(stage, str(exc)[:200])


def default_model(provider: str) -> str:
    """The model a provider will actually use by default (local: the best installed vision model)."""
    if provider == "ollama":
        try:
            vlm = registry.vlms.create("ollama")
            return vlm.resolve_model(vlm.model, explicit=False)
        except Exception as exc:  # noqa: BLE001
            log.info("local model resolution skipped: %s", exc)
    return PROFILES[provider].default_model


def _model(req: RunRequest) -> str:
    if not req.vlm:
        return ""
    return req.vlm.model or default_model(req.vlm.provider)


def _make_vlm(req: RunRequest, run: AnalysisRun):
    registry.load_plugins()
    try:
        vlm = registry.vlms.create(req.vlm.provider)
    except Exception as exc:  # noqa: BLE001
        run.warnings.append(f"AI provider unavailable: {exc}")
        return None
    run.data_egress = sorted(set(run.data_egress) | {req.vlm.provider})
    run.pipeline.append(StageConfig(stage="vlm", impl=req.vlm.provider, params={"model": _model(req)}))
    return vlm


def _make_decider(key: str, run: AnalysisRun):
    registry.load_plugins()
    for candidate in (key, "rules"):
        try:
            decider = registry.deciders.create(candidate)
            if candidate != key:
                run.warnings.append(f"System One '{key}' unavailable; using the rules baseline")
            return decider
        except Exception as exc:  # noqa: BLE001
            if candidate == "rules":
                run.warnings.append(f"No System One decider available: {exc}")
    return None


# Deciders that call a hosted provider (text state; openai-decisions can also take rim crops).
DECIDER_PROVIDERS = {
    "openai-decisions": "openai",
    "adapter:claude": "claude",
    "adapter:openai": "openai",
    "adapter:gemini": "gemini",
    "jev": "typesafe",
}


def _run_cap(req: RunRequest) -> float:
    settings = get_settings()
    return min(req.max_cost_usd or settings.run_cost_cap_usd, settings.run_cost_cap_usd)


def _per_call_estimates(req: RunRequest, clip: ClipMeta) -> dict[str, float]:
    """Expected USD per summarize / adjudicate call, for pre-call cap checks (conservative fallback 0.05)."""
    out = {"summarize": 0.05, "adjudicate": 0.05}
    if not req.vlm:
        return out
    try:
        for line in estimate(req, clip).breakdown:
            if line.stage == "summarize":
                out["summarize"] = max(line.usd, 0.0)
            elif line.stage == "adjudicate":
                n = max(
                    1, min(req.max_escalations, max(1, round(max(1, int(clip.master.duration_s / 12)) * 0.4)))
                )
                out["adjudicate"] = max(line.usd / n, 0.0)
    except Exception as exc:  # noqa: BLE001
        log.info("per-call estimates unavailable: %s", exc)
    return out


def _record_vlm(run: AnalysisRun, store: Store, result: Any, purpose: str, guard: Any) -> None:
    import secrets

    call_id = f"{purpose}-{secrets.token_hex(3)}"
    raw_path = None
    if getattr(result, "raw", None) is not None:
        try:
            raw_path = store.save_raw(run.run_id, call_id, result.raw)
        except Exception:  # noqa: BLE001
            raw_path = None
    usage = getattr(result, "usage", None)
    rec = VLMCallRecord(
        call_id=call_id,
        provider=result.provider,
        model=result.model_id,
        purpose=purpose,  # type: ignore[arg-type]
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        stop_reason=result.stop_reason,
        raw_path=raw_path,
        error=result.error,
    )
    run.vlm_calls.append(rec)
    if result.cost_usd:
        run.metrics.cost_usd = round(run.metrics.cost_usd + result.cost_usd, 6)
        if guard is not None:
            guard.add(result.cost_usd)
    if usage is not None:
        run.metrics.tokens_in += getattr(usage, "input_tokens", 0) or 0
        run.metrics.tokens_out += getattr(usage, "output_tokens", 0) or 0


def _triage(run: AnalysisRun, decider: Any, detections: Any, clip: ClipMeta) -> None:
    if decider is None:
        return
    try:
        from hoops.decide.questions import triage_state
        from hoops.events.escalate import triage

        per_min = len(clip.scene_cuts_s) / max(clip.master.duration_s / 60.0, 1e-6)
        counts = detections["cls"].value_counts().to_dict()
        frames = max(int(detections["frame"].nunique()), 1)
        state = triage_state(
            scene_cuts="many" if per_min > 6 else "some" if per_min > 0.5 else "none",
            orientation="portrait" if clip.master.height > clip.master.width else "landscape",
            rim="visible"
            if counts.get("rim", 0) > frames * 0.5
            else "sometimes"
            if counts.get("rim")
            else "not found",
            players="many" if (counts.get("player", 0) + counts.get("person", 0)) / frames > 6 else "few",
            ball="often" if counts.get("ball", 0) > frames * 0.3 else "rarely",
        )
        footage, record = triage(decider, state)
        run.decisions.append(record)
        run.clip.footage_type, run.clip.footage_type_source = footage, "system_one"
    except Exception as exc:  # noqa: BLE001
        run.warnings.append(f"Footage triage skipped: {exc}")


def _auto_rim(run: AnalysisRun, proxy: Path, detections: Any) -> tuple[float, float, float, float] | None:
    try:
        from hoops.frames import frames_at
        from hoops.perception.rim_finder import pick_active_rim, suggest_rims

        dur = run.clip.proxy.duration_s
        suggestions = []
        for _, rgb in frames_at(proxy, [dur * 0.1, dur * 0.5]):
            suggestions += suggest_rims(rgb)
        best = pick_active_rim(suggestions, detections)
    except Exception as exc:  # noqa: BLE001
        run.warnings.append(
            f"Couldn't find the rim automatically ({exc}); mark it on the clip to detect shots"
        )
        return None
    if best is None:
        run.warnings.append("No rim found: mark the rim on the clip to detect shots")
        return None
    box = tuple(round(v, 1) for v in best.box)
    run.warnings = [w for w in run.warnings if not w.startswith("No stable rim box")]
    run.pipeline.append(
        StageConfig(
            stage="rim",
            impl="owlv2:zero-shot",
            license="Apache-2.0",
            params={"box": list(box), "score": best.score, "auto": True},
        )
    )
    run.warnings.append(
        "Rim found automatically: if the rim box in the video is on the wrong hoop, mark the rim and re-run"
    )
    return box  # type: ignore[return-value]


def _rim_crops_fn(proxy: Path):
    from hoops.frames import crop, frames_at, to_jpeg

    def images_for(cand: Any) -> list[bytes]:
        times = [cand.t_key_s - 0.2, cand.t_key_s, cand.t_key_s + 0.25, cand.t_key_s + 0.5]
        out = []
        for _, rgb in frames_at(proxy, times):
            box = getattr(cand, "rim_box", None)
            img = crop(rgb, box, pad=1.5, min_size=160) if box else rgb
            out.append(to_jpeg(img, max_px=448, quality=80))
        return out

    return images_for


def _cv_context(run: AnalysisRun) -> str:
    def mmss(t: float) -> str:
        return f"{int(t // 60)}:{t % 60:04.1f}"

    events = [
        {
            "t": mmss(e.t_key_s or e.t_start_s),
            "type": e.type.value,
            "result": e.shot.result if e.shot else None,
            "team": e.team_id,
            "status": e.status,
        }
        for e in run.events
        if e.status != "rejected"
    ]
    stats = [
        {"team": s.team.team_id, "color": s.team.color_hex, "fgm": s.stats.fgm, "fga": s.stats.fga}
        for s in run.team_summaries
    ]
    return json.dumps(
        {
            "duration_s": round(run.clip.master.duration_s, 1),
            "footage_type": run.clip.footage_type,
            "teams": stats,
            "events_from_cv": events,
            "tracked_people": len(run.tracks),
        }
    )


def _court_maps(run: AnalysisRun, req: RunRequest, store: Store, detections: Any, proxy: Path) -> list[str]:
    from hoops import charts
    from hoops.frames import frames_at

    out_dir = store.run_dir(run.run_id) / "charts"
    out_dir.mkdir(exist_ok=True)
    paths: list[str] = []
    shots = [e for e in run.events if e.shot and e.status != "rejected" and e.shot.image_x is not None]
    people = None
    if detections is not None and len(detections):
        ppl = detections[detections["cls"].isin(["player", "person"])]
        people = np.stack([(ppl["x1"] + ppl["x2"]) / 2, ppl["y2"]], axis=1) if len(ppl) else None

    calibration = None
    if req.court_points and len(req.court_points) >= 4:
        from hoops.perception.court import Calibration

        calibration = Calibration.from_points(
            [(c.x, c.y) for c in req.court_points], [c.name for c in req.court_points], court="nba"
        )
        if not calibration.is_valid:
            run.warnings.append(
                "Court calibration looks off (high reprojection error); showing image-space maps"
            )
            calibration = None

    if calibration is not None:
        pts = np.array([[e.shot.image_x, e.shot.image_y] for e in shots]) if shots else np.zeros((0, 2))
        court_xy = calibration.to_court(pts) if len(pts) else pts
        for e, (x, y) in zip(shots, court_xy, strict=False):
            e.shot.court_x_ft, e.shot.court_y_ft = float(x), float(y)
        made = [
            (float(x), float(y), e.shot.result == "made") for e, (x, y) in zip(shots, court_xy, strict=False)
        ]
        paths.append(store.rel(charts.shot_chart(made, out_dir / "shot_chart.png", title="Shot chart")))
        if people is not None:
            paths.append(
                store.rel(charts.court_heatmap(calibration.to_court(people), out_dir / "heatmap.png"))
            )
    else:
        if run.clip.footage_type not in ("fixed_camera", "broadcast", "unknown"):
            run.warnings.append("Court map needs a fixed or broadcast camera; showing an image-space heatmap")
        poster = frames_at(proxy, [run.clip.proxy.duration_s * 0.1])
        if people is not None and poster:
            paths.append(store.rel(charts.image_heatmap(people, poster[0][1], out_dir / "image_heatmap.png")))
    return paths
