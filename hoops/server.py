"""FastAPI backend: implements docs/api.md and serves the web UI (web/dist).

Single-user POC: jobs and sessions live in memory; clips and runs are persisted under data/.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from hoops import __version__, registry
from hoops.config import OUTPUTS, PROFILES, ROOT, get_settings
from hoops.decide.base import PRESETS
from hoops.jobs import JobContext, JobManager
from hoops.pipeline import RunRequest, estimate, run_analysis, stages_for
from hoops.schemas import AnalysisRun, ClipMeta, Event, EventType, Narrative, Provenance, ShotInfo
from hoops.store import Store

log = logging.getLogger(__name__)

WEB_DIST = ROOT / "web" / "dist"
COOKIE = "hoops_session"
SESSION_TTL_S = 7 * 24 * 3600
CHUNK_SIZE = 8 * 1024 * 1024
OPEN_PATHS = {"/api/health", "/api/session", "/api/login", "/api/logout"}
OUTPUT_LABELS = {
    "annotated_video": "Annotated video",
    "events": "Event log + stats",
    "summary": "AI summary & Q&A",
    "court_map": "Court map / heatmap",
}
FOOTAGE_TYPES = ["unknown", "broadcast", "fixed_camera", "handheld", "highlights", "practice"]


class LoginBody(BaseModel):
    password: str


class UploadInit(BaseModel):
    filename: str
    size_bytes: int
    content_type: str = "application/octet-stream"


class ClipPatch(BaseModel):
    footage_type: Literal["unknown", "broadcast", "fixed_camera", "handheld", "highlights", "practice"]


class RecomputeBody(BaseModel):
    threshold_preset: str


class ReviewBody(BaseModel):
    action: Literal["accept", "reject"]
    note: str | None = None


class AddEventBody(BaseModel):
    t_s: float
    type: Literal["made_shot", "missed_shot", "shot_attempt"]
    team_id: str | None = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class AskBody(BaseModel):
    question: str
    provider: str | None = None
    model: str | None = None
    history: list[HistoryTurn] = []


def create_app(store: Store | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ai_press_hoops", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json"
    )
    app.state.store = store or Store()
    app.state.jobs = JobManager()
    app.state.secret = secrets.token_bytes(32)
    app.state.spend = 0.0
    app.state.models_cache = {}
    app.state.default_models = {}
    app.state.provider_health = {}  # provider key -> human reason it can't be used right now
    app.state.login_failures = {}
    registry.load_plugins()
    threading.Thread(target=_health_checks, args=(app,), daemon=True).start()

    def st() -> Store:
        return app.state.store

    # ------------------------------------------------------------------ auth
    def auth_required() -> bool:
        return bool(settings.key("app_password"))

    def make_token(user: str) -> str:
        payload = f"{user}|{int(time.time()) + SESSION_TTL_S}".encode()
        sig = hmac.new(app.state.secret, payload, hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(payload).decode() + "." + sig

    def session_user(request: Request) -> str | None:
        token = request.cookies.get(COOKIE)
        if not token or "." not in token:
            return None
        b64, sig = token.rsplit(".", 1)
        try:
            payload = base64.urlsafe_b64decode(b64.encode())
        except Exception:  # noqa: BLE001
            return None
        good = hmac.new(app.state.secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sig):
            return None
        user, exp = payload.decode().rsplit("|", 1)
        return user if int(exp) > time.time() else None

    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        path = request.url.path
        protected = (path.startswith("/api/") and path not in OPEN_PATHS) or path.startswith("/media/")
        if protected and auth_required() and session_user(request) is None:
            return JSONResponse({"detail": "Login required"}, status_code=401)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/session")
    def session(request: Request) -> dict:
        user = session_user(request)
        return {
            "authenticated": (not auth_required()) or user is not None,
            "auth_required": auth_required(),
            "user": user,
        }

    @app.post("/api/login")
    def login(body: LoginBody, request: Request, response: Response) -> dict:
        ip = request.client.host if request.client else "?"
        fails = app.state.login_failures.get(ip, 0)
        if fails >= 5:
            time.sleep(min(5.0, 0.5 * fails))  # slow brute force from one address
        expected = settings.key("app_password")
        if expected is None:
            return {"ok": True}
        if not hmac.compare_digest(body.password.encode(), expected.encode()):
            app.state.login_failures[ip] = fails + 1
            raise HTTPException(401, "Wrong password")
        app.state.login_failures.pop(ip, None)
        response.set_cookie(
            COOKIE, make_token(settings.app_user), max_age=SESSION_TTL_S, httponly=True, samesite="lax"
        )
        return {"ok": True}

    @app.post("/api/logout")
    def logout(response: Response) -> dict:
        response.delete_cookie(COOKIE)
        return {"ok": True}

    # ------------------------------------------------------------------ config
    def entry_view(e: registry.Entry, **extra: Any) -> dict:
        return {
            "key": e.key,
            "label": e.label,
            "available": e.unavailable_reason() is None,
            "reason": e.unavailable_reason(),
            "license": e.license,
            "notes": e.notes,
            **extra,
        }

    def provider_models(key: str, default: str) -> list[str]:
        cached = app.state.models_cache.get(key)
        if cached is None:
            app.state.models_cache[key] = [default]
            threading.Thread(target=_fetch_models, args=(app, key, default), daemon=True).start()
            return [default]
        return cached

    @app.get("/api/config")
    def config() -> dict:
        vlm_entries = {e.key: e for e in registry.vlms.entries()}
        providers = []
        for key, prof in PROFILES.items():
            entry = vlm_entries.get(key)
            reason = entry.unavailable_reason() if entry else "adapter not installed"
            reason = reason or app.state.provider_health.get(key)
            default = app.state.default_models.get(key, prof.default_model)
            providers.append(
                {
                    "key": key,
                    "label": prof.label,
                    "available": reason is None,
                    "reason": reason,
                    "native_video": prof.native_video,
                    "audio": prof.audio,
                    "local": prof.local,
                    "sends_data_to": prof.sends_data_to,
                    "default_model": default,
                    "models": provider_models(key, default) if reason is None else [default],
                }
            )
        try:
            from hoops.reason.prompts import list_presets

            prompt_presets = list_presets()
        except Exception:  # noqa: BLE001
            prompt_presets = [
                {"key": "recap", "label": "Recap"},
                {"key": "play_by_play", "label": "Play-by-play"},
                {"key": "coach", "label": "Coach's notes"},
                {"key": "custom", "label": "Custom"},
            ]
        return {
            "version": __version__,
            "providers": providers,
            "detectors": [
                entry_view(e) for e in registry.detectors.entries() if not e.key.startswith("fake:")
            ],
            "trackers": [entry_view(e) for e in registry.trackers.entries()],
            "deciders": [
                entry_view(e, supports_images=bool(e.extra.get("supports_images")))
                for e in registry.deciders.entries()
            ],
            "threshold_presets": [
                {"key": k, "label": k.title(), "accept": p.accept, "reject": p.reject, "note": p.note}
                for k, p in PRESETS.items()
            ],
            "prompt_presets": prompt_presets,
            "outputs": [{"key": k, "label": OUTPUT_LABELS[k]} for k in OUTPUTS],
            "footage_types": FOOTAGE_TYPES,
            "courts": _courts(),
            "caps": {
                "run_cost_cap_usd": settings.run_cost_cap_usd,
                "session_cost_cap_usd": settings.session_cost_cap_usd,
                "max_upload_mb": settings.max_upload_mb,
                "max_clip_seconds": settings.max_clip_seconds,
            },
            "spend": {"session_usd": round(app.state.spend, 4)},
        }

    # ------------------------------------------------------------------ uploads
    uploads_dir = st().root / "uploads"

    @app.post("/api/uploads")
    def upload_init(body: UploadInit) -> dict:
        if body.size_bytes > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(400, f"File is larger than {settings.max_upload_mb} MB")
        upload_id = f"u_{secrets.token_hex(6)}"
        d = uploads_dir / upload_id
        d.mkdir(parents=True)
        (d / "meta.json").write_text(body.model_dump_json())
        return {"upload_id": upload_id, "chunk_size_bytes": CHUNK_SIZE}

    def upload_dir(upload_id: str) -> Path:
        if not re.fullmatch(r"u_[0-9a-f]{12}", upload_id):
            raise HTTPException(404, "Unknown upload")
        d = uploads_dir / upload_id
        if not d.exists():
            raise HTTPException(404, "Unknown upload")
        return d

    @app.put("/api/uploads/{upload_id}/chunks/{index}")
    async def upload_chunk(upload_id: str, index: int, request: Request) -> dict:
        d = upload_dir(upload_id)
        if index < 0 or index > 100_000:
            raise HTTPException(400, "Bad chunk index")
        data = await request.body()
        if len(data) > CHUNK_SIZE * 2:
            raise HTTPException(400, "Chunk too large")
        (d / f"{index:06d}.part").write_bytes(data)
        received = sum(p.stat().st_size for p in d.glob("*.part"))
        return {"received_bytes": received}

    @app.post("/api/uploads/{upload_id}/complete")
    def upload_complete(upload_id: str) -> dict:
        d = upload_dir(upload_id)
        meta = UploadInit.model_validate_json((d / "meta.json").read_text())
        parts = sorted(d.glob("*.part"))
        if not parts:
            raise HTTPException(400, "No data uploaded")
        suffix = Path(meta.filename).suffix.lower()[:8] or ".mp4"
        src = d / f"source{suffix}"
        with src.open("wb") as out:
            for part in parts:
                with part.open("rb") as f:
                    shutil.copyfileobj(f, out)
                part.unlink()
        if src.stat().st_size != meta.size_bytes:
            log.warning("upload %s size mismatch: %s != %s", upload_id, src.stat().st_size, meta.size_bytes)

        def work(ctx: JobContext) -> None:
            from hoops.ingest import ingest

            ctx.start("ingest")
            clip = ingest(src, original_filename=meta.filename, store=st(), progress=ctx.callback("ingest"))
            ctx.set(clip_id=clip.clip_id)
            ctx.done("ingest", clip.normalization.summary or "Ready")
            shutil.rmtree(d, ignore_errors=True)

        job = app.state.jobs.submit("ingest", work, stages=["ingest"])
        return {"job_id": job.job_id}

    # ------------------------------------------------------------------ jobs
    def get_job(job_id: str):
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return get_job(job_id).model_dump(mode="json", exclude={"version"})

    @app.get("/api/jobs/{job_id}/stream")
    async def job_stream(job_id: str) -> StreamingResponse:
        job = get_job(job_id)

        async def events():
            last, last_beat = -1, time.monotonic()
            while True:
                if job.version != last:
                    last = job.version
                    yield f"event: job\ndata: {job.model_dump_json(exclude={'version'})}\n\n"
                    if job.terminal:
                        return
                elif time.monotonic() - last_beat > 15:
                    last_beat = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> dict:
        job = app.state.jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job.model_dump(mode="json", exclude={"version"})

    # ------------------------------------------------------------------ clips
    def clip_urls(clip: ClipMeta) -> dict:
        base = f"/media/clips/{clip.clip_id}"
        return {
            "master": f"/media/{clip.master.path}",
            "proxy": f"/media/{clip.proxy.path}",
            "poster": f"{base}/poster.jpg",
        }

    def load_clip(clip_id: str) -> ClipMeta:
        if not re.fullmatch(r"[0-9a-f]{8,64}", clip_id):
            raise HTTPException(404, "Unknown clip")
        clip = st().load_clip(clip_id)
        if clip is None:
            raise HTTPException(404, "Unknown clip")
        return clip

    @app.get("/api/clips")
    def clips() -> list[dict]:
        out = []
        for p in sorted(st().clips_dir.glob("*/clip.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                c = ClipMeta.model_validate_json(p.read_text())
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping unreadable clip %s: %s", p.parent.name, exc)
                continue
            out.append(
                {
                    "clip_id": c.clip_id,
                    "original_filename": c.original_filename,
                    "duration_s": c.master.duration_s,
                    "footage_type": c.footage_type,
                    "poster_url": clip_urls(c)["poster"],
                    "created_at": c.created_at.isoformat(),
                }
            )
        return out

    @app.get("/api/clips/{clip_id}")
    def clip(clip_id: str) -> dict:
        c = load_clip(clip_id)
        return {**c.model_dump(mode="json"), "urls": clip_urls(c)}

    @app.patch("/api/clips/{clip_id}")
    def clip_patch(clip_id: str, body: ClipPatch) -> dict:
        c = load_clip(clip_id)
        c.footage_type, c.footage_type_source = body.footage_type, "human"
        st().save_clip(c)
        return {**c.model_dump(mode="json"), "urls": clip_urls(c)}

    # ------------------------------------------------------------------ runs
    def run_view(run: AnalysisRun) -> dict:
        def media(rel: str | None) -> str | None:
            return f"/media/{rel}" if rel else None

        urls = {
            "annotated": media(run.annotated_video_path),
            "master": media(run.clip.master.path),
            "proxy": media(run.clip.proxy.path),
            "captions": media(run.captions_path),
            "court_maps": [media(p) for p in run.court_map_paths],
            "poster": f"/media/clips/{run.clip.clip_id}/poster.jpg",
            "download": f"/api/runs/{run.run_id}/download",
        }
        return {**run.model_dump(mode="json"), "urls": urls}

    def load_run(run_id: str) -> AnalysisRun:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", run_id):
            raise HTTPException(404, "Unknown run")
        try:
            return st().load_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Unknown run") from exc

    @app.post("/api/estimate")
    def estimate_route(req: RunRequest) -> dict:
        return estimate(req, load_clip(req.clip_id)).model_dump()

    @app.post("/api/runs")
    def create_run(req: RunRequest) -> dict:
        clip_meta = load_clip(req.clip_id)
        if req.vlm and req.vlm.provider not in PROFILES:
            raise HTTPException(400, f"Unknown provider {req.vlm.provider}")
        if req.vlm and not PROFILES[req.vlm.provider].local and req.vlm.provider not in req.egress_ok:
            raise HTTPException(
                400, f"Allow sending frames to {PROFILES[req.vlm.provider].sends_data_to} first"
            )
        est = estimate(req, clip_meta)
        if app.state.spend + est.estimate_usd > settings.session_cost_cap_usd:
            raise HTTPException(402, f"This would pass the ${settings.session_cost_cap_usd:.2f} session cap")
        run_id = Store.new_run_id()

        def work(ctx: JobContext) -> None:
            run = run_analysis(req, ctx, st(), run_id=run_id)
            app.state.spend += run.metrics.cost_usd
            for call in run.vlm_calls:
                if call.error and (why := billing_problem(call.provider, call.error)):
                    app.state.provider_health[call.provider] = why

        job = app.state.jobs.submit(
            "analyze", work, stages=stages_for(req), clip_id=req.clip_id, run_id=run_id
        )
        return {"run_id": run_id, "job_id": job.job_id, "estimate_usd": est.estimate_usd}

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return [r.model_dump(mode="json") for r in st().list_runs()]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        return run_view(load_run(run_id))

    @app.get("/api/runs/{run_id}/download")
    def download(run_id: str) -> FileResponse:
        load_run(run_id)
        run_dir = st().runs_dir / run_id
        tmp = Path(tempfile.mkstemp(suffix=".zip")[1])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in run_dir.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(run_dir))
        return FileResponse(tmp, media_type="application/zip", filename=f"hoops-{run_id}.zip")

    @app.post("/api/runs/{run_id}/recompute")
    def recompute(run_id: str, body: RecomputeBody) -> dict:
        from hoops.events.escalate import recompute as do_recompute
        from hoops.events.stats import compute_stats

        r = load_run(run_id)
        preset = PRESETS.get(body.threshold_preset)
        if preset is None:
            raise HTTPException(400, "Unknown preset")
        r.events = do_recompute(r.events, preset)
        r.team_summaries = compute_stats(r.events, r.teams)
        st().save_run(r)
        return {
            "events": [e.model_dump(mode="json") for e in r.events],
            "team_summaries": [s.model_dump(mode="json") for s in r.team_summaries],
        }

    @app.post("/api/runs/{run_id}/events/{event_id}/review")
    def review(run_id: str, event_id: str, body: ReviewBody) -> dict:
        r = load_run(run_id)
        ev = next((e for e in r.events if e.event_id == event_id), None)
        if ev is None:
            raise HTTPException(404, "Unknown event")
        ev.review = "accepted" if body.action == "accept" else "rejected"
        ev.status = "confirmed" if body.action == "accept" else "rejected"
        if body.note:
            ev.description = body.note
        _refresh_stats(r)
        st().save_run(r)
        return ev.model_dump(mode="json")

    @app.post("/api/runs/{run_id}/events")
    def add_event(run_id: str, body: AddEventBody) -> dict:
        r = load_run(run_id)
        n = sum(1 for e in r.events if e.provenance.source == "human") + 1
        result = {"made_shot": "made", "missed_shot": "missed"}.get(body.type, "unknown")
        ev = Event(
            event_id=f"h{n}",
            type=EventType(body.type),
            t_start_s=body.t_s,
            t_key_s=body.t_s,
            team_id=body.team_id,
            shot=ShotInfo(result=result),
            confidence=1.0,
            confidence_kind="human",
            status="confirmed",
            review="accepted",
            provenance=Provenance(source="human", producer="review", stage="review"),
        )
        r.events.append(ev)
        r.events.sort(key=lambda e: e.t_key_s or e.t_start_s)
        _refresh_stats(r)
        st().save_run(r)
        return ev.model_dump(mode="json")

    @app.post("/api/runs/{run_id}/ask")
    def ask(run_id: str, body: AskBody) -> dict:
        from hoops.reason.qa import answer

        r = load_run(run_id)
        provider = body.provider or next((c.provider for c in r.vlm_calls), None)
        vlm = None
        if provider:
            if provider not in PROFILES:
                raise HTTPException(400, f"Unknown provider {provider}")
            try:
                vlm = registry.vlms.create(provider)
            except Exception as exc:
                raise HTTPException(503, str(exc)) from exc
            if not PROFILES[provider].local:
                r.data_egress = sorted(
                    set(r.data_egress) | {provider}
                )  # explicit pick in the Ask panel = consent
        decider = None
        for key in [s.impl for s in r.pipeline if s.stage == "decide"] + ["rules"]:
            try:
                decider = registry.deciders.create(key)
                break
            except Exception as exc:  # noqa: BLE001
                log.info("decider %s unavailable for Q&A routing: %s", key, exc)
        history = []
        turns = list(body.history)
        for i in range(0, len(turns) - 1, 2):
            if turns[i].role == "user" and turns[i + 1].role == "assistant":
                history.append((turns[i].text, turns[i + 1].text))
        if app.state.spend >= settings.session_cost_cap_usd:
            vlm = None  # cap reached: answer from the event log only
        res = answer(
            r, body.question, vlm=vlm, decider=decider, history=history, video=st().abs(r.clip.master.path)
        )
        app.state.spend += res.cost_usd or 0.0
        r.narratives.append(
            Narrative(
                kind="qa",
                text=res.answer,
                question=body.question,
                cited_times_s=[c.get("t_s") for c in res.citations if c.get("t_s") is not None],
                provenance=Provenance(
                    source="vlm" if res.provider else "rule",
                    producer=f"{res.provider}:{res.model}",
                    stage="qa",
                    params={"route": res.route},
                ),
            )
        )
        r.metrics.cost_usd = round(r.metrics.cost_usd + (res.cost_usd or 0.0), 6)
        st().save_run(r)
        return {
            "answer": res.answer,
            "route": res.route,
            "citations": res.citations,
            "cost_usd": res.cost_usd,
            "provider": res.provider,
            "model": res.model,
            "latency_ms": res.latency_ms,
            "decision": res.decision,
        }

    @app.get("/api/compare")
    def compare(a: str, b: str) -> dict:
        ra, rb = load_run(a), load_run(b)
        return {"a": run_view(ra), "b": run_view(rb), "agreement": _agreement(ra, rb)}

    # ------------------------------------------------------------------ media + web UI
    @app.get("/media/{path:path}")
    def media(path: str) -> FileResponse:
        root = st().root.resolve()
        target = (root / path).resolve()
        allowed = (root / "clips", root / "runs")
        if not any(target.is_relative_to(a.resolve()) for a in allowed) or not target.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(target)

    @app.get("/{full_path:path}", include_in_schema=False)
    def web(full_path: str):
        if full_path.startswith(("api/", "media/")):
            raise HTTPException(404, "Not found")
        index = WEB_DIST / "index.html"
        if not index.exists():
            return HTMLResponse(_NO_BUILD_HTML, status_code=200)
        candidate = (WEB_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(WEB_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)

    return app


def _courts() -> list[dict]:
    """Court specs + calibration landmark names (the UI's point picker must use these exact names)."""
    try:
        from hoops.perception.court import COURTS, landmarks
    except Exception:  # noqa: BLE001
        return []
    out = []
    for key, spec in COURTS.items():
        marks = [
            {
                "name": name,
                "label": name.replace("_", " ").capitalize(),
                "x_ft": x,
                "y_ft": y,
                "filmed_half": not name.startswith("other_"),
            }
            for name, (x, y) in landmarks(key).items()
            if name != "rim"
        ]
        out.append(
            {
                "key": key,
                "label": spec.label,
                "length_ft": spec.length_ft,
                "width_ft": spec.width_ft,
                "landmarks": marks,
            }
        )
    return out


def _refresh_stats(run: AnalysisRun) -> None:
    try:
        from hoops.events.stats import compute_stats

        run.team_summaries = compute_stats(run.events, run.teams)
    except Exception as exc:  # noqa: BLE001
        log.warning("stats refresh failed: %s", exc)


def _family(e: Event) -> str:
    return (
        "shot"
        if e.type.value in ("shot_attempt", "made_shot", "missed_shot", "three_pointer", "free_throw")
        else e.type.value
    )


def _agreement(a: AnalysisRun, b: AnalysisRun, tol_s: float = 1.0) -> dict:
    ea = [e for e in a.events if e.status != "rejected"]
    eb = [e for e in b.events if e.status != "rejected"]
    used: set[int] = set()
    pairs = []
    for x in ea:
        tx = x.t_key_s or x.t_start_s
        best, best_d = None, tol_s + 1e-9
        for j, y in enumerate(eb):
            if j in used or _family(x) != _family(y):
                continue
            d = abs(tx - (y.t_key_s or y.t_start_s))
            if d <= best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((x, eb[best]))
    shot_pairs = [(x, y) for x, y in pairs if x.shot and y.shot]
    agree = [x.shot.result == y.shot.result for x, y in shot_pairs]
    return {
        "matched": len(pairs),
        "only_a": len(ea) - len(pairs),
        "only_b": len(eb) - len(pairs),
        "outcome_agreement": round(sum(agree) / len(agree), 3) if agree else None,
    }


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
        return f"{PROFILES[provider].label} account is out of credits (add billing at {where})"
    return None


def _health_checks(app: FastAPI) -> None:
    """Free startup checks: Anthropic token counting reports an empty balance; Ollama's installed models."""
    settings = get_settings()
    if settings.key("anthropic_api_key"):
        try:
            import anthropic

            anthropic.Anthropic(
                api_key=settings.key("anthropic_api_key"), max_retries=0
            ).messages.count_tokens(
                model=PROFILES["claude"].default_model, messages=[{"role": "user", "content": "hi"}]
            )
        except Exception as exc:  # noqa: BLE001
            if why := billing_problem("claude", str(exc)):
                app.state.provider_health["claude"] = why
    try:
        vlm = registry.vlms.create("ollama")
        resolved = vlm.resolve_model(vlm.model, explicit=False)
        app.state.default_models["ollama"] = resolved
        if not vlm._installed:
            app.state.provider_health["ollama"] = "Ollama isn't running (start the Ollama app)"
    except Exception as exc:  # noqa: BLE001
        log.info("ollama health check skipped: %s", exc)


def _fetch_models(app: FastAPI, key: str, default: str) -> None:
    try:
        vlm = registry.vlms.create(key)
        models = vlm.list_models() or [default]
        if default in models:
            models = [default] + [m for m in models if m != default]
        app.state.models_cache[key] = models[:40]
    except Exception as exc:  # noqa: BLE001
        log.info("model list for %s unavailable: %s", key, exc)


_NO_BUILD_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>ai_press_hoops</title>
<style>body{font-family:system-ui;background:#0b1020;color:#e8ecf5;display:grid;place-items:center;height:100vh;margin:0}
code{background:#1b2440;padding:2px 6px;border-radius:6px}</style></head><body><div>
<h1>ai_press_hoops API is running</h1><p>The web UI hasn't been built yet. Run:</p>
<p><code>npm --prefix web install &amp;&amp; npm --prefix web run build</code></p>
<p>API docs: <a style="color:#ff8a3d" href="/api/docs">/api/docs</a></p></div></body></html>"""


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
