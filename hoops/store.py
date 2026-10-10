"""On-disk layout. Append-only runs; clips cached by content hash.

data/
  clips/<clip_id>/{source.<ext>, probe.json, clip.json, master.mp4, proxy_720p.mp4}
  runs/<run_id>/{run.json, detections.parquet, annotated.mp4, events.vtt, sheets/, crops/, raw_vlm/, charts/}
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from hoops.config import get_settings
from hoops.schemas import AnalysisRun, ClipMeta

log = logging.getLogger(__name__)


class RunSummary(BaseModel):
    run_id: str
    created_at: datetime
    status: str
    clip_id: str
    filename: str | None
    footage_type: str
    outputs: list[str]
    n_events: int
    cost_usd: float
    providers: list[str]


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_settings().data_dir
        self.clips_dir = self.root / "clips"
        self.runs_dir = self.root / "runs"

    # ------------------------------------------------------------------ paths
    def clip_dir(self, clip_id: str) -> Path:
        d = self.clips_dir / clip_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path | str) -> str:
        """Path relative to the data root (what we store in JSON)."""
        return str(Path(path).resolve().relative_to(self.root.resolve()))

    def abs(self, rel_path: str) -> Path:
        return self.root / rel_path

    @staticmethod
    def new_run_id() -> str:
        return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    # ------------------------------------------------------------------ clips
    def save_clip(self, clip: ClipMeta) -> None:
        _atomic_write(self.clip_dir(clip.clip_id) / "clip.json", clip.model_dump_json(indent=2))

    def load_clip(self, clip_id: str) -> ClipMeta | None:
        p = self.clips_dir / clip_id / "clip.json"
        return ClipMeta.model_validate_json(p.read_text()) if p.exists() else None

    # ------------------------------------------------------------------ runs
    def save_run(self, run: AnalysisRun) -> Path:
        p = self.run_dir(run.run_id) / "run.json"
        _atomic_write(p, run.model_dump_json(indent=2))
        return p

    def load_run(self, run_id: str) -> AnalysisRun:
        return AnalysisRun.model_validate_json((self.runs_dir / run_id / "run.json").read_text())

    def save_raw(self, run_id: str, name: str, payload: object) -> str:
        """Save a raw provider response (dict/list/str) under raw_vlm/; returns the data-relative path."""
        d = self.run_dir(run_id) / "raw_vlm"
        d.mkdir(exist_ok=True)
        p = d / f"{name}.json"
        _atomic_write(p, payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str))
        return self.rel(p)

    def list_runs(self) -> list[RunSummary]:
        out: list[RunSummary] = []
        if not self.runs_dir.exists():
            return out
        for p in sorted(self.runs_dir.glob("*/run.json"), reverse=True):
            try:
                r = AnalysisRun.model_validate_json(p.read_text())
            except Exception as exc:  # noqa: BLE001 - a corrupt run must not break the list
                log.warning("skipping unreadable run %s: %s", p.parent.name, exc)
                continue
            out.append(
                RunSummary(
                    run_id=r.run_id,
                    created_at=r.created_at,
                    status=r.status,
                    clip_id=r.clip.clip_id,
                    filename=r.clip.original_filename,
                    footage_type=r.clip.footage_type,
                    outputs=list(r.outputs_requested),
                    n_events=len(r.events),
                    cost_usd=r.metrics.cost_usd,
                    providers=sorted({c.provider for c in r.vlm_calls}),
                )
            )
        return out


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
