from __future__ import annotations

from hoops.events.stats import compute_stats, events_digest
from hoops.schemas import Event, EventType, Provenance, ShotInfo, TeamRef


def _ev(eid, etype, *, team=None, status="confirmed", points=None, result="unknown", kind="unknown", t=1.0):
    return Event(
        event_id=eid,
        type=etype,
        t_start_s=t,
        t_key_s=t,
        team_id=team,
        shot=ShotInfo(points=points, result=result, kind=kind),
        confidence=0.9,
        confidence_kind="heuristic",
        status=status,
        provenance=Provenance(source="rule", producer="t", stage="decide"),
    )


def test_compute_stats():
    teams = [TeamRef(team_id="A", name="White"), TeamRef(team_id="B")]
    events = [
        _ev("e1", EventType.made_shot, team="A", result="made"),
        _ev("e2", EventType.made_shot, team="A", result="made", points=3),
        _ev("e3", EventType.missed_shot, team="A", points=3, result="missed"),
        _ev("e4", EventType.shot_attempt, team="B", status="needs_review"),
        _ev("e5", EventType.made_shot, team="B", status="rejected", result="made"),  # ignored
        _ev("e6", EventType.made_shot, team="Z", result="made"),  # unknown team id -> unknown bucket
        _ev("e7", EventType.made_shot, result="made", kind="free_throw"),
        _ev("e8", EventType.rebound, team="A"),  # not a shot
    ]
    a, b, unknown = compute_stats(events, teams)
    assert a.team.team_id == "A" and b.team.team_id == "B" and unknown.team.team_id == "unknown"
    assert (a.stats.fga, a.stats.fgm, a.stats.tpa, a.stats.tpm, a.stats.pts) == (3, 2, 2, 1, 5)
    assert a.derived_from_event_ids == ["e1", "e2", "e3"]
    assert (b.stats.fga, b.stats.fgm, b.stats.pts) == (1, 0, 0) and b.derived_from_event_ids == ["e4"]
    assert (unknown.stats.fga, unknown.stats.fgm, unknown.stats.fta, unknown.stats.ftm) == (1, 1, 1, 1)
    assert unknown.stats.pts == 3 and unknown.derived_from_event_ids == ["e6", "e7"]


def test_empty_and_digest():
    [unknown] = compute_stats([], [])
    assert unknown.stats.fga == 0 and unknown.derived_from_event_ids == []
    events = [
        _ev("e1", EventType.made_shot, team="A", result="made", t=65.25),
        _ev("e2", EventType.missed_shot, status="rejected", t=3.0),
    ]
    events[0].description = "Made jumper"
    assert events_digest(events) == "1:05.2 made jumper, team A, confirmed"
    assert events_digest([]) == "no events"
