from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hoops.eval import (
    EvalReport,
    auroc,
    by_footage_type,
    labels_from_run,
    load_labels,
    match,
    predicted_shots,
    report_markdown,
    run_eval,
    score_run,
    sweep,
    wilson,
)
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
from hoops.store import Store

ROOT = Path(__file__).resolve().parent.parent


def _ev(eid, etype, t, conf=0.9, **kw):
    return Event(
        event_id=eid,
        type=etype,
        t_start_s=t,
        confidence=conf,
        confidence_kind="heuristic",
        provenance=kw.pop("provenance", Provenance(source="rule", producer="test", stage="shots")),
        **kw,
    )


def _run(events, clip_id="clip1", run_id="r1", footage="fixed_camera"):
    asset = VideoAsset(
        path="x.mp4", width=1280, height=720, fps="30/1", fps_float=30, n_frames=1, duration_s=60
    )
    clip = ClipMeta(
        clip_id=clip_id,
        source_size_bytes=1,
        master=asset,
        proxy=asset,
        normalization=NormalizationReport(),
        footage_type=footage,
    )
    return AnalysisRun(run_id=run_id, clip=clip, events=events, status="done")


LABELS = {
    "clip_id": "clip1",
    "footage_type": "fixed_camera",
    "events": [
        {"t_s": 10.0, "type": "made_shot"},
        {"t_s": 20.0, "type": "missed_shot"},
        {"t_s": 30.0, "type": "made_shot"},
        {"t_s": 40.0, "type": "shot_attempt", "note": "outcome hidden by a player"},
    ],
}


def test_match_greedy_one_to_one():
    matches, fp, fn = match([0.0, 5.0, 10.4], [0.5, 10.0, 20.0], tol_s=1.0)
    assert matches == [(0, 0), (2, 1)] and fp == [1] and fn == [2]
    matches, fp, fn = match([1.0, 1.1], [1.2], tol_s=1.0)  # closest pair wins; one gold, one match
    assert matches == [(1, 0)] and fp == [0] and fn == []
    assert match([2.0], [1.0], tol_s=1.0)[0] == [(0, 0)]  # boundary is inclusive
    assert match([], [1.0]) == ([], [], [0])


def test_wilson_known_values():
    lo, hi = wilson(27, 30)
    assert lo == pytest.approx(0.7438, abs=1e-3) and hi == pytest.approx(0.9654, abs=1e-3)
    lo, hi = wilson(90, 100)
    assert lo == pytest.approx(0.8256, abs=1e-3) and hi == pytest.approx(0.9448, abs=1e-3)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_auroc():
    assert auroc([0.9, 0.8, 0.7, 0.6], [True, True, False, False]) == 1.0
    assert auroc([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == 0.75
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(auroc([0.9], [True]))


def test_predicted_shots_collapse_linked_events():
    events = [
        _ev("a1", EventType.shot_attempt, 10.0, t_key_s=10.2, features={"make_score": 0.4}),
        _ev("m1", EventType.made_shot, 10.5, related_event_ids=["a1"], features={"make_score": 0.9}),
        _ev("a2", EventType.shot_attempt, 20.0, shot=ShotInfo(result="missed")),
        _ev("x", EventType.rebound, 21.0),
        _ev("r", EventType.made_shot, 50.0, status="rejected"),
    ]
    shots = predicted_shots(events)
    assert [(s.t_s, s.made) for s in shots] == [(10.2, True), (20.0, False)]
    assert shots[0].features["make_score"] == 0.9 and sorted(shots[0].event_ids) == ["a1", "m1"]
    assert len(predicted_shots(events, include_rejected=True)) == 3


def test_score_run():
    events = [
        _ev("e1", EventType.made_shot, 10.3, conf=0.95),
        _ev("e2", EventType.missed_shot, 19.4, conf=0.8),
        _ev("e3", EventType.made_shot, 31.6, conf=0.4),  # 1.6 s off: a false alarm and a miss
        _ev("e4", EventType.shot_attempt, 40.2, conf=0.7),
        _ev("e5", EventType.made_shot, 55.0, conf=0.3),  # phantom make
    ]
    rep = score_run(_run(events), LABELS)
    assert (rep.attempts.tp, rep.attempts.fp, rep.attempts.fn) == (3, 2, 1)
    assert rep.attempts.precision == pytest.approx(0.6) and rep.attempts.recall == pytest.approx(0.75)
    assert (rep.makes.tp, rep.makes.fp, rep.makes.fn) == (1, 2, 1)
    assert rep.outcome_n == 2 and rep.outcome_accuracy == 1.0
    assert rep.confidence_auroc == 1.0  # every right call outscored every wrong one
    assert rep.false_alarms_s == [31.6, 55.0] and rep.missed_s == [30.0]
    assert rep.attempts.precision_ci == pytest.approx(wilson(3, 5))
    md = report_markdown(rep)
    assert "| Shot attempts | 60%" in md and "False alarms at 0:31.6, 0:55.0" in md
    assert EvalReport.model_validate_json(rep.model_dump_json()) == rep
    assert json.loads(rep.model_dump_json())["outcome_accuracy"] == 1.0


def test_sweep_trades_precision_for_recall():
    events = [
        _ev("e1", EventType.shot_attempt, 10.0, features={"make_score": 0.9}),
        _ev("e2", EventType.shot_attempt, 20.0, features={"make_score": 0.6}, status="rejected"),
        _ev("e3", EventType.shot_attempt, 30.0, features={"make_score": 0.5}),
        _ev("e4", EventType.shot_attempt, 40.0, features={"make_score": 0.2}),
    ]
    curve = {t: (p, r) for t, p, r in sweep(events, LABELS, thresholds=[0.0, 0.55, 0.95])}
    assert curve[0.0] == (0.5, 1.0)  # 4 kept, 2 are the labeled makes (10 s, 30 s)
    assert curve[0.55] == (0.5, 0.5)
    assert curve[0.95] == (1.0, 0.0)
    attempts = sweep(events, LABELS, score_key="make_score", target="attempt", thresholds=[0.0])
    assert attempts == [(0.0, 1.0, 1.0)]


def test_by_footage_type_and_markdown_list():
    a = score_run(_run([_ev("e1", EventType.made_shot, 10.0)]), LABELS)
    b = score_run(
        _run([], clip_id="clip2", run_id="r2"), {**LABELS, "clip_id": "clip2", "footage_type": "broadcast"}
    )
    groups = by_footage_type([a, b])
    assert set(groups) == {"broadcast", "fixed_camera"} and groups["broadcast"].attempts.fn == 4
    md = report_markdown([a, b])
    assert "### broadcast" in md and "### all footage: 2 clip(s)" in md
    assert "No labeled clips" in report_markdown([])


def test_labels_from_reviewed_run():
    human = Provenance(source="human", producer="friend", stage="review")
    run = _run(
        [
            _ev("e1", EventType.made_shot, 10.0, review="accepted"),
            _ev("e2", EventType.missed_shot, 20.0),  # unreviewed: not gold
            _ev("e3", EventType.shot_attempt, 30.0, provenance=human),
        ]
    )
    labels = labels_from_run(run)
    assert labels["events"] == [{"t_s": 10.0, "type": "made_shot"}, {"t_s": 30.0, "type": "shot_attempt"}]


def test_example_labels_and_run_eval(data_dir, tmp_path):
    example = load_labels(ROOT / "eval" / "labels" / "example.json")
    assert example["events"] and example["clip_id"]
    store = Store()
    store.save_run(_run([_ev("e1", EventType.made_shot, 10.0)]))
    (tmp_path / "clip1.json").write_text(json.dumps(LABELS))
    (tmp_path / "other.json").write_text(json.dumps({**LABELS, "clip_id": "nope"}))
    (tmp_path / "bad.json").write_text("{}")
    reports, notes = run_eval(tmp_path, store)
    assert [r.clip_id for r in reports] == ["clip1"] and len(notes) == 2


@pytest.mark.parametrize(
    "bad", [{}, {"clip_id": "x"}, {"clip_id": "x", "events": [{"t_s": -1, "type": "made_shot"}]},
            {"clip_id": "x", "events": [{"t_s": 1, "type": "dunk"}]}],
)  # fmt: skip
def test_bad_labels(bad):
    with pytest.raises(ValueError):
        score_run(_run([]), bad)
