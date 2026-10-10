from __future__ import annotations

from hoops.captions import vtt_time, write_vtt
from hoops.schemas import Event, EventType, Provenance, ShotInfo


def _event(eid, etype, t, **kw):
    return Event(
        event_id=eid,
        type=etype,
        t_start_s=t,
        confidence=0.9,
        confidence_kind="heuristic",
        provenance=Provenance(source="rule", producer="test", stage="shots"),
        **kw,
    )


def test_vtt_time():
    assert vtt_time(0) == "00:00:00.000"
    assert vtt_time(75.5) == "00:01:15.500"
    assert vtt_time(3725.0429) == "01:02:05.043"


def test_write_vtt(tmp_path):
    events = [
        _event("e2", EventType.missed_shot, 9.0, team_id="B", status="needs_review"),
        _event("e1", EventType.made_shot, 3.0, t_key_s=4.25, team_id="A", shot=ShotInfo(kind="jumper")),
        _event("e3", EventType.made_shot, 12.0, status="rejected"),
        _event("e4", EventType.shot_attempt, 20.0, shot=ShotInfo(result="made", points=3, kind="unknown")),
        _event("e5", EventType.rebound, 21.0, review="rejected"),
    ]
    out = write_vtt(events, tmp_path / "runs" / "events.vtt")
    text = out.read_text()
    assert text.startswith("WEBVTT\n\n")
    blocks = text.strip().split("\n\n")[1:]
    assert [b.splitlines()[0] for b in blocks] == ["e1", "e2", "e4"]
    assert blocks[0].splitlines()[1:] == ["00:00:04.250 --> 00:00:05.750", "MADE · jumper · Team A"]
    assert blocks[1].splitlines()[2] == "MISS · Team B (check)"
    assert blocks[2].splitlines()[2] == "MADE 3PT"


def test_write_vtt_empty(tmp_path):
    assert write_vtt([], tmp_path / "e.vtt").read_text().startswith("WEBVTT")
