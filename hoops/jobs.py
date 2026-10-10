"""In-process job runner for long work (ingest, analysis) with per-stage progress.

Heavy analysis runs one at a time (MPS-bound); ingest jobs get their own small pool. Job state is
in memory (runs themselves are persisted by the pipeline), which is enough for a single-user POC.
"""

from __future__ import annotations

import logging
import secrets
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

STAGES: list[tuple[str, str]] = [
    ("ingest", "Preparing video"),
    ("detect", "Finding players & ball"),
    ("track", "Tracking"),
    ("shots", "Spotting shots"),
    ("decide", "System One decisions"),
    ("adjudicate", "AI review of close calls"),
    ("summarize", "AI summary"),
    ("court", "Court map"),
    ("render", "Annotated video"),
]
STAGE_LABELS = dict(STAGES)

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
StageStatus = Literal["pending", "running", "done", "skipped", "failed"]


class StageState(BaseModel):
    key: str
    label: str
    status: StageStatus = "pending"
    progress: float = 0.0


class Job(BaseModel):
    job_id: str
    kind: Literal["ingest", "analyze"]
    status: JobStatus = "queued"
    progress: float = 0.0
    stage: str | None = None
    message: str = ""
    clip_id: str | None = None
    run_id: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stages: list[StageState] = Field(default_factory=list)
    version: int = 0  # bumps on every change (SSE uses it to emit only updates)

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "failed", "cancelled")


class Cancelled(Exception):
    pass


class JobContext:
    """Handed to job functions: report progress per stage, check for cancellation."""

    def __init__(self, manager: JobManager, job: Job):
        self._m = manager
        self.job = job

    @property
    def cancelled(self) -> bool:
        return self._m._cancel_flags.get(self.job.job_id, False)

    def check_cancel(self) -> None:
        if self.cancelled:
            raise Cancelled()

    def start(self, stage: str, message: str | None = None) -> None:
        self._m._update(
            self.job,
            stage=stage,
            stage_status="running",
            stage_progress=0.0,
            message=message or STAGE_LABELS.get(stage, stage),
        )

    def progress(self, stage: str, fraction: float, message: str | None = None) -> None:
        self._m._update(
            self.job, stage=stage, stage_status="running", stage_progress=fraction, message=message
        )

    def done(self, stage: str, message: str | None = None) -> None:
        self._m._update(self.job, stage=stage, stage_status="done", stage_progress=1.0, message=message)

    def skip(self, stage: str, reason: str | None = None) -> None:
        self._m._update(
            self.job,
            stage=stage,
            stage_status="skipped",
            stage_progress=0.0,
            message=reason,
            move_cursor=False,
        )

    def fail(self, stage: str, message: str) -> None:
        self._m._update(self.job, stage=stage, stage_status="failed", message=message, move_cursor=False)

    def callback(self, stage: str) -> Callable[[float, str], None]:
        """A ``progress(fraction, message)`` callable for library code; also checks cancellation."""

        def cb(fraction: float, message: str = "") -> None:
            self.check_cancel()
            self.progress(stage, fraction, message or None)

        return cb

    def set(self, **fields: object) -> None:
        self._m._update(self.job, fields=fields)


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cancel_flags: dict[str, bool] = {}
        self._analyze_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyze")
        self._ingest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit(
        self,
        kind: Literal["ingest", "analyze"],
        fn: Callable[[JobContext], None],
        *,
        stages: list[str],
        clip_id: str | None = None,
        run_id: str | None = None,
    ) -> Job:
        job = Job(
            job_id=f"j_{secrets.token_hex(4)}",
            kind=kind,
            clip_id=clip_id,
            run_id=run_id,
            stages=[StageState(key=k, label=STAGE_LABELS.get(k, k)) for k in stages],
        )
        self.jobs[job.job_id] = job
        pool = self._analyze_pool if kind == "analyze" else self._ingest_pool
        pool.submit(self._run, job, fn)
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and not job.terminal:
            self._cancel_flags[job_id] = True
            if job.status == "queued":
                self._update(job, fields={"status": "cancelled", "finished_at": _now()}, message="Cancelled")
        return job

    # ------------------------------------------------------------------ internals
    def _run(self, job: Job, fn: Callable[[JobContext], None]) -> None:
        if self._cancel_flags.get(job.job_id):
            return
        ctx = JobContext(self, job)
        self._update(job, fields={"status": "running", "started_at": _now()})
        try:
            fn(ctx)
        except Cancelled:
            self._update(job, fields={"status": "cancelled", "finished_at": _now()}, message="Cancelled")
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            log.error("job %s failed: %s\n%s", job.job_id, exc, traceback.format_exc())
            if job.stage:
                self._update(job, stage=job.stage, stage_status="failed", move_cursor=False)
            self._update(
                job,
                fields={"status": "failed", "error": _friendly(exc), "finished_at": _now()},
                message="Something went wrong",
            )
            return
        for s in job.stages:
            if s.status == "pending":
                s.status = "skipped"
        self._update(job, fields={"status": "done", "progress": 1.0, "finished_at": _now()}, message="Done")

    def _update(
        self,
        job: Job,
        *,
        stage: str | None = None,
        stage_status: StageStatus | None = None,
        stage_progress: float | None = None,
        message: str | None = None,
        move_cursor: bool = True,
        fields: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            if stage is not None:
                st = next((s for s in job.stages if s.key == stage), None)
                if st is None:
                    st = StageState(key=stage, label=STAGE_LABELS.get(stage, stage))
                    job.stages.append(st)
                if stage_status:
                    st.status = stage_status
                if stage_progress is not None:
                    st.progress = max(0.0, min(1.0, float(stage_progress)))
                if move_cursor:
                    job.stage = stage
            if message is not None:
                job.message = message
            for k, v in (fields or {}).items():
                setattr(job, k, v)
            if job.stages and "progress" not in (fields or {}):
                active = [s for s in job.stages if s.status != "skipped"]
                if active:
                    job.progress = sum(1.0 if s.status == "done" else s.progress for s in active) / len(
                        active
                    )
            job.version += 1


def _now() -> datetime:
    return datetime.now(UTC)


def _friendly(exc: Exception) -> str:
    """Error text safe to show a friend: no local paths (which include the macOS user name)."""
    from pathlib import Path

    from hoops.config import get_settings

    text = str(exc) or type(exc).__name__
    for path, alias in (
        (str(get_settings().data_dir.resolve()), "<data>"),
        (str(get_settings().data_dir), "<data>"),
        (str(Path.home()), "~"),
    ):
        text = text.replace(path, alias)
    return text if len(text) < 500 else text[:500] + "…"
