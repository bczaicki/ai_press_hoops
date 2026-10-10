"""Box-score stats computed in code from the event log. VLMs explain; they never count.

Rejected events are ignored. needs_review events still count (the UI badges them), so Strict <->
Eager changes the stats only through rejections. Free throws (event type, shot kind, or points == 1)
count as FTA/FTM. A made field goal is worth 3 when ``shot.points == 3``, and 2 otherwise,
including when the distance is unknown (no court map). 3PA/3PM only count when points are known.
"""

from __future__ import annotations

from collections.abc import Sequence

from hoops.schemas import Event, EventType, ShotInfo, StatLine, TeamRef, TeamSummary

UNKNOWN_TEAM = TeamRef(team_id="unknown", name="Unknown team")
FIELD_GOAL_TYPES = {
    EventType.made_shot,
    EventType.missed_shot,
    EventType.shot_attempt,
    EventType.three_pointer,
}


def compute_stats(events: Sequence[Event], teams: Sequence[TeamRef]) -> list[TeamSummary]:
    """Per-team FGA/FGM, 3PA/3PM, FTA/FTM, and points, plus an 'unknown' team bucket (always last)."""
    buckets: dict[str, tuple[TeamRef, StatLine, list[str]]] = {t.team_id: (t, StatLine(), []) for t in teams}
    buckets.setdefault(UNKNOWN_TEAM.team_id, (UNKNOWN_TEAM, StatLine(), []))
    for e in events:
        if e.status == "rejected" or (e.type not in FIELD_GOAL_TYPES and e.type != EventType.free_throw):
            continue
        _, s, ids = buckets.get(e.team_id or UNKNOWN_TEAM.team_id) or buckets[UNKNOWN_TEAM.team_id]
        shot = e.shot or ShotInfo()
        made = e.type == EventType.made_shot or shot.result == "made"
        if e.type == EventType.free_throw or shot.kind == "free_throw" or shot.points == 1:
            s.fta += 1
            s.ftm += int(made)
            s.pts += int(made)
        else:
            three = shot.points == 3 or e.type == EventType.three_pointer
            s.fga += 1
            s.tpa += int(three)
            if made:
                s.fgm += 1
                s.tpm += int(three)
                s.pts += 3 if three else 2
        ids.append(e.event_id)
    return [TeamSummary(team=t, stats=s, derived_from_event_ids=ids) for t, s, ids in buckets.values()]


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def events_digest(events: Sequence[Event], *, limit: int = 60) -> str:
    """One short line per non-rejected event, for Q&A routing and claim-check states."""
    lines = []
    for e in sorted(events, key=lambda e: e.t_key_s if e.t_key_s is not None else e.t_start_s):
        if e.status == "rejected":
            continue
        t = e.t_key_s if e.t_key_s is not None else e.t_start_s
        team = f"team {e.team_id}" if e.team_id else "team unknown"
        what = e.description or e.type.value.replace("_", " ")
        lines.append(f"{_clock(t)} {what.lower()}, {team}, {e.status.replace('_', ' ')}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more"]
    return "\n".join(lines) if lines else "no events"
