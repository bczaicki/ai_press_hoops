"""WebVTT captions for the event log, so the HTML5 player shows "MADE · jumper · Team A" at each event."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hoops.schemas import Event, EventType

LABELS = {
    EventType.shot_attempt: "SHOT",
    EventType.made_shot: "MADE",
    EventType.missed_shot: "MISS",
    EventType.three_pointer: "3PT",
    EventType.free_throw: "FREE THROW",
    EventType.rebound: "REBOUND",
    EventType.assist: "ASSIST",
    EventType.turnover: "TURNOVER",
    EventType.steal: "STEAL",
    EventType.block: "BLOCK",
    EventType.foul: "FOUL",
    EventType.possession_change: "POSSESSION",
    EventType.timeout: "TIMEOUT",
    EventType.other: "EVENT",
}


def vtt_time(t_s: float) -> str:
    ms = max(0, round(t_s * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def cue_text(event: Event) -> str:
    head = LABELS.get(event.type, "EVENT")
    shot = event.shot
    if shot is not None:
        if event.type == EventType.shot_attempt and shot.result != "unknown":
            head = "MADE" if shot.result == "made" else "MISS"
        if shot.points == 3 and event.type != EventType.three_pointer:
            head += " 3PT"
    parts = [head]
    if shot is not None and shot.kind not in ("unknown", "free_throw"):
        parts.append(shot.kind)
    if event.team_id:
        parts.append(f"Team {event.team_id}")
    text = " · ".join(parts)
    if event.status == "needs_review":
        text += " (check)"
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_vtt(events: Sequence[Event], path: Path, *, duration_s: float = 1.5) -> Path:
    """Write one cue per non-rejected event, starting at its key time (else its start)."""
    keep = [e for e in events if e.status != "rejected" and e.review != "rejected"]
    keep.sort(key=lambda e: e.t_key_s if e.t_key_s is not None else e.t_start_s)
    lines = ["WEBVTT", ""]
    for e in keep:
        t = e.t_key_s if e.t_key_s is not None else e.t_start_s
        cue_id = e.event_id.replace("-->", "->").replace("\n", " ")
        lines += [cue_id, f"{vtt_time(t)} --> {vtt_time(t + duration_s)}", cue_text(e), ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
