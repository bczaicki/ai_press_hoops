from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hoops import config
from hoops.jobs import JobManager
from hoops.schemas import (
    AnalysisRun,
    ClipMeta,
    Event,
    EventType,
    NormalizationReport,
    Provenance,
    ShotInfo,
    VideoAsset,
)
from hoops.server import _agreement, create_app
from hoops.store import Store


def _asset(path: str) -> VideoAsset:
    return VideoAsset(
        path=path, width=1280, height=720, fps="30", fps_float=30.0, n_frames=90, duration_s=3.0
    )


def _clip(clip_id: str = "0123456789abcdef") -> ClipMeta:
    return ClipMeta(
        clip_id=clip_id,
        source_size_bytes=1,
        master=_asset(f"clips/{clip_id}/master.mp4"),
        proxy=_asset(f"clips/{clip_id}/proxy_720p.mp4"),
        normalization=NormalizationReport(),
    )


def _event(eid: str, t: float, result: str = "made") -> Event:
    return Event(
        event_id=eid,
        type=EventType.made_shot if result == "made" else EventType.missed_shot,
        t_start_s=t,
        t_key_s=t,
        shot=ShotInfo(result=result),
        confidence=0.9,
        confidence_kind="heuristic",
        provenance=Provenance(source="rule", producer="t", stage="shots"),
    )


@pytest.fixture
def client(data_dir, monkeypatch):
    monkeypatch.delenv("HOOPS_APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    config.get_settings.cache_clear()
    with TestClient(create_app(Store(data_dir))) as c:
        yield c


def test_health_session_config(client):
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/session").json() == {"authenticated": True, "auth_required": False, "user": None}
    cfg = client.get("/api/config").json()
    assert {p["key"] for p in cfg["providers"]} >= {"claude", "gemini", "openai", "openrouter", "ollama"}
    assert [o["key"] for o in cfg["outputs"]] == ["annotated_video", "events", "summary", "court_map"]
    assert {p["key"] for p in cfg["threshold_presets"]} == {"strict", "balanced", "eager"}


def test_auth_required_when_password_set(data_dir, monkeypatch):
    monkeypatch.setenv("HOOPS_APP_PASSWORD", "hunter2")
    config.get_settings.cache_clear()
    with TestClient(create_app(Store(data_dir))) as c:
        assert c.get("/api/config").status_code == 401
        assert c.get("/media/clips/x/master.mp4").status_code == 401
        assert c.get("/api/session").json()["auth_required"] is True
        assert c.post("/api/login", json={"password": "nope"}).status_code == 401
        assert c.post("/api/login", json={"password": "hunter2"}).status_code == 200
        assert c.get("/api/config").status_code == 200
        assert c.get("/api/session").json()["authenticated"] is True
    config.get_settings.cache_clear()


def test_chunked_upload_assembles_file(client, data_dir):
    payload = b"x" * 1000
    up = client.post("/api/uploads", json={"filename": "clip.mov", "size_bytes": len(payload)}).json()
    uid = up["upload_id"]
    assert client.put(f"/api/uploads/{uid}/chunks/0", content=payload[:600]).json()["received_bytes"] == 600
    assert client.put(f"/api/uploads/{uid}/chunks/1", content=payload[600:]).json()["received_bytes"] == 1000
    job_id = client.post(f"/api/uploads/{uid}/complete").json()["job_id"]
    for _ in range(100):  # not a real video: the ingest job must fail cleanly, not crash the server
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "failed" and job["error"]


def test_media_is_confined_to_data_dir(client, data_dir):
    (data_dir / "secret.txt").write_text("no")
    (data_dir / "clips" / "abc").mkdir(parents=True)
    (data_dir / "clips" / "abc" / "poster.jpg").write_bytes(b"\xff\xd8jpg")
    assert client.get("/media/clips/abc/poster.jpg").status_code == 200
    assert client.get("/media/secret.txt").status_code == 404
    assert client.get("/media/clips/../secret.txt").status_code == 404


def test_run_view_review_and_add_event(client, data_dir):
    store = Store(data_dir)
    run = AnalysisRun(run_id=Store.new_run_id(), clip=_clip(), events=[_event("e1", 1.0)])
    store.save_run(run)
    view = client.get(f"/api/runs/{run.run_id}").json()
    assert view["urls"]["download"].endswith("/download") and view["events"][0]["event_id"] == "e1"
    pytest.importorskip("hoops.events.stats")
    ev = client.post(f"/api/runs/{run.run_id}/events/e1/review", json={"action": "reject"}).json()
    assert ev["status"] == "rejected"
    added = client.post(f"/api/runs/{run.run_id}/events", json={"t_s": 2.0, "type": "made_shot"}).json()
    assert added["provenance"]["source"] == "human" and added["status"] == "confirmed"
    assert client.get(f"/api/runs/{run.run_id}/download").headers["content-type"] == "application/zip"


def test_agreement_matches_within_tolerance():
    a = AnalysisRun(run_id="a", clip=_clip(), events=[_event("a1", 1.0), _event("a2", 5.0, "missed")])
    b = AnalysisRun(run_id="b", clip=_clip(), events=[_event("b1", 1.4), _event("b2", 9.0)])
    assert _agreement(a, b) == {"matched": 1, "only_a": 1, "only_b": 1, "outcome_agreement": 1.0}


def test_spa_fallback_without_build(client):
    r = client.get("/some/deep/link")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/api/nope").status_code == 404


def test_job_manager_progress_and_cancel():
    jm = JobManager()

    def work(ctx):
        ctx.start("detect")
        ctx.progress("detect", 0.5, "half")
        ctx.done("detect")
        ctx.skip("render")

    job = jm.submit("analyze", work, stages=["detect", "render"])
    for _ in range(100):
        if job.terminal:
            break
        time.sleep(0.01)
    assert job.status == "done" and job.progress == 1.0
    assert [s.status for s in job.stages] == ["done", "skipped"]

    def slow(ctx):
        for _ in range(200):
            ctx.check_cancel()
            time.sleep(0.01)

    job2 = jm.submit("analyze", slow, stages=["detect"])
    time.sleep(0.05)
    jm.cancel(job2.job_id)
    for _ in range(200):
        if job2.terminal:
            break
        time.sleep(0.01)
    assert job2.status == "cancelled"


def test_billing_problem_messages():
    from hoops.server import billing_problem

    msg = billing_problem(
        "claude", "Error code: 400 - Your credit balance is too low to access the Anthropic API"
    )
    assert msg and "out of credits" in msg and "console.anthropic.com" in msg
    assert billing_problem("openai", "429 insufficient_quota") is not None
    assert billing_problem("openai", "500 internal server error") is None
