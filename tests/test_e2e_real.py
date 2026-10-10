"""End-to-end on real CC BY footage (scripts/fetch_samples.py). Slow: downloads/loads RF-DETR weights."""

from __future__ import annotations

import pytest

from hoops.config import ROOT
from hoops.store import Store
from hoops.video_io import is_browser_playable

SAMPLE = ROOT / "data" / "samples" / "plos_shot_01.ogv"
RIM_BOX = [768.0, 240.0, 824.0, 266.0]  # proxy px, the hoop under the backboard in this clip


@pytest.mark.slow
def test_real_clip_end_to_end(data_dir):
    if not SAMPLE.exists():
        pytest.skip("run: uv run python scripts/fetch_samples.py")
    ingest = pytest.importorskip("hoops.ingest").ingest
    pytest.importorskip("hoops.events.escalate")
    from hoops.pipeline import RunRequest, run_analysis

    store = Store(data_dir)
    clip = ingest(SAMPLE, original_filename=SAMPLE.name, store=store)
    assert clip.proxy.height == 720 and clip.master.codec == "h264"
    req = RunRequest(
        clip_id=clip.clip_id,
        outputs=["annotated_video", "events", "court_map"],
        detector="rfdetr:coco",
        tracker="bytetrack",
        decider="rules",
        rim_box=RIM_BOX,
    )
    run = run_analysis(req, None, store)
    assert run.status == "done", run.warnings
    assert run.tracks, "the shooter should be tracked"
    assert run.annotated_video_path and is_browser_playable(store.abs(run.annotated_video_path))
    assert run.detections_path and store.abs(run.detections_path).exists()
    assert store.load_run(run.run_id).run_id == run.run_id
