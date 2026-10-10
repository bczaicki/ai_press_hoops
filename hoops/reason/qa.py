"""Ask: routed Q&A over one analysis run.

Routes (a System One ``Choice`` when a decider is given, else keywords):
- ``stats`` and ``find_moments`` are answered in code from ``run.events`` / ``run.team_summaries``: exact, instant,
  free. Rejected events are ignored.
- ``explain_play`` sends the referenced (or best-matching) event window to the VLM as labeled sheets plus event
  context; with no clear target it uses whole-clip overview sheets.
- ``summary`` reuses a saved narrative when there is one, else asks the VLM for a recap.
- ``unsupported`` explains what the clip data can answer (and that we never identify people).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hoops.decide.base import Choice
from hoops.reason.base import AnalyzeRequest, VisionLLM
from hoops.reason.prompts import load_prompt, system_prompt
from hoops.reason.sheets import TILE_RE, clock, overview_images, resolve_ref, window_images
from hoops.schemas import AnalysisRun, Event, EventType

ROUTES = ("stats", "find_moments", "explain_play", "summary", "unsupported")
ROUTE_QUESTION = Choice(
    instructions="Which kind of answer does this question about a basketball clip need?",
    criteria={
        "stats": "Counts, totals, scores or percentages: how many shots, points, field-goal percentage.",
        "find_moments": "Finding when things happened: when, show me, list every make or miss.",
        "explain_play": "Explaining or describing a specific play or moment: what happened at 0:12, why it missed.",
        "summary": "An overall recap or summary of the clip.",
        "unsupported": "Anything else, including naming or identifying real people.",
    },
)
SHOT_TYPES = {
    EventType.shot_attempt,
    EventType.made_shot,
    EventType.missed_shot,
    EventType.three_pointer,
    EventType.free_throw,
}
UNSUPPORTED_WORDS = (
    "real name", "name of", "his name", "her name", "their name", "identify", "recognize", "recognise",
    "face", "who is that", "who's that", "who is the player", "who is number", "who is #",
)  # fmt: skip
SUMMARY_WORDS = (
    "summar",
    "recap",
    "overview",
    "tl;dr",
    "tldr",
    "highlights",
    "sum up",
    "how did the game go",
)
FIND_WORDS = (
    "when ", "when's", "what time", "at what point", "show me", "find ", "where in the", "timestamp",
    "list every", "list all", "list the", "jump to", "which moments", "every time",
)  # fmt: skip
STATS_WORDS = (
    "how many", "count", "number of", "total", "score", "points", "percentage", "%", "fg", "field goal",
    "shooting", "stats", "statistic", "who won", "winning", "leading", "efficien",
)  # fmt: skip
TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2}(?:\.\d+)?)\b|\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b", re.IGNORECASE
)

UNSUPPORTED_ANSWER = (
    "I can't answer that from this clip's data. I can tell you shot counts and percentages, find when shots, "
    'rebounds or turnovers happened, explain a specific play (for example "what happened at 0:12?"), or '
    "summarize the clip. I never identify people by face or name; players are described by jersey color and "
    "number."
)


@dataclass
class QAAnswer:
    answer: str
    route: str
    citations: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    decision: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------- event helpers


def event_time(e: Event) -> float:
    return e.t_key_s if e.t_key_s is not None else e.t_start_s


def live_events(run: AnalysisRun) -> list[Event]:
    return sorted(
        (e for e in run.events if e.status != "rejected" and e.review != "rejected"),
        key=event_time,
    )


def shot_result(e: Event) -> str:
    if e.shot is not None and e.shot.result != "unknown":
        return e.shot.result
    return {EventType.made_shot: "made", EventType.missed_shot: "missed"}.get(e.type, "unknown")


def team_name(run: AnalysisRun, team_id: str | None) -> str:
    for t in run.teams:
        if t.team_id == team_id:
            return t.name or f"Team {team_id}"
    return f"Team {team_id}" if team_id else "unknown team"


def event_label(run: AnalysisRun, e: Event) -> str:
    if e.type in SHOT_TYPES:
        result = {"made": "Made", "missed": "Missed"}.get(shot_result(e), "")
        if e.type == EventType.free_throw or (e.shot and e.shot.kind == "free_throw"):
            what = "free throw"
        elif e.type == EventType.three_pointer or (e.shot and e.shot.points == 3):
            what = "3-pointer"
        elif e.shot and e.shot.kind not in ("unknown", "free_throw"):
            what = e.shot.kind
        else:
            what = "shot"
        what = f"{result} {what}" if result else f"{what.capitalize()} attempt"
    else:
        what = e.type.value.replace("_", " ").capitalize()
    team = f" ({team_name(run, e.team_id)})" if e.team_id else ""
    return f"{what}{team}, {clock(event_time(e))}"


def cite(run: AnalysisRun, e: Event) -> dict:
    return {"t_s": round(event_time(e), 2), "event_id": e.event_id, "label": event_label(run, e)}


def event_brief(run: AnalysisRun, e: Event) -> dict:
    return {
        "event_id": e.event_id,
        "label": event_label(run, e),
        "t_s": round(event_time(e), 2),
        "window_s": [round(e.t_start_s, 2), round(e.t_end_s, 2) if e.t_end_s is not None else None],
        "status": e.status,
        "confidence": round(e.confidence, 2),
        "description": e.description,
    }


def run_context(run: AnalysisRun, events: Sequence[Event] | None = None, limit: int = 60) -> str:
    """Compact CV context JSON for prompts."""
    evs = list(events) if events is not None else live_events(run)
    ctx: dict[str, Any] = {
        "clip_seconds": round(run.clip.master.duration_s, 2),
        "footage_type": run.clip.footage_type,
        "teams": [{"team_id": t.team_id, "name": t.name, "color": t.color_hex} for t in run.teams],
        "team_stats": [
            {"team": team_name(run, s.team.team_id), **s.stats.model_dump(exclude_defaults=True)}
            for s in run.team_summaries
        ],
        "events": [event_brief(run, e) for e in evs[:limit]],
    }
    return json.dumps(ctx, separators=(",", ":"))


def parse_time(question: str) -> float | None:
    m = TIME_RE.search(question)
    if not m:
        return None
    if m.group(1) is not None:
        return int(m.group(1)) * 60 + float(m.group(2))
    return float(m.group(3))


# ---------------------------------------------------------------------------- routing


def keyword_route(question: str) -> str:
    q = f" {question.lower()} "
    if any(w in q for w in UNSUPPORTED_WORDS):
        return "unsupported"
    if any(w in q for w in SUMMARY_WORDS):
        return "summary"
    if any(w in q for w in FIND_WORDS):
        return "find_moments"
    if any(w in q for w in STATS_WORDS):
        return "stats"
    return "explain_play"


def route_question(
    question: str, decider: Any = None, history: Sequence[tuple[str, str]] = ()
) -> tuple[str, dict | None]:
    """(route, decision record or None). Falls back to keywords when the decider fails or answers off-menu."""
    if decider is None:
        return keyword_route(question), None
    state = {"question": question}
    if history:
        state["previous_question"] = history[-1][0]
    try:
        res = decider.decide(state, {"route": ROUTE_QUESTION})
    except Exception as exc:  # noqa: BLE001 - routing must never break Ask
        return keyword_route(question), {
            "decider": getattr(decider, "key", None),
            "error": str(exc),
            "fallback": "keywords",
        }
    ans = (res.answers or {}).get("route") if not res.error else None
    decision = {
        "decider": res.decider,
        "model": res.model,
        "route_confidence": getattr(ans, "confidence", None),
        "probabilities": getattr(ans, "probabilities", None),
        "cost_usd": res.cost_usd,
        "latency_ms": res.latency_ms,
    }
    if ans is not None and str(ans.value) in ROUTES:
        return str(ans.value), decision
    decision.update(
        error=res.error or f"unexpected route {getattr(ans, 'value', None)!r}", fallback="keywords"
    )
    return keyword_route(question), decision


# ---------------------------------------------------------------------------- code answers


def _filters(run: AnalysisRun, question: str) -> list[Callable[[Event], bool]]:
    q = f" {question.lower()} "

    def kind(k: str) -> Callable[[Event], bool]:
        return lambda e: e.shot is not None and e.shot.kind == k

    def typ(*ts: EventType) -> Callable[[Event], bool]:
        return lambda e: e.type in ts

    table: list[tuple[tuple[str, ...], Callable[[Event], bool]]] = [
        (("three", "3-point", "3 point", "3pt", "triple"), lambda e: e.type == EventType.three_pointer or (e.shot is not None and e.shot.points == 3)),
        (("free throw", "foul shot"), lambda e: e.type == EventType.free_throw or (e.shot is not None and e.shot.kind == "free_throw")),
        (("dunk",), kind("dunk")),
        (("layup", "lay-up", "lay up"), kind("layup")),
        (("jumper", "jump shot"), kind("jumper")),
        ((" miss", "brick"), lambda e: e.type in SHOT_TYPES and shot_result(e) == "missed"),
        ((" made", " make", "score", "bucket", "basket", "went in"), lambda e: e.type in SHOT_TYPES and shot_result(e) == "made"),
        (("rebound", "board"), typ(EventType.rebound)),
        (("turnover",), typ(EventType.turnover)),
        (("steal",), typ(EventType.steal)),
        (("block",), typ(EventType.block)),
        ((" foul",), typ(EventType.foul)),
        (("assist",), typ(EventType.assist)),
        (("timeout",), typ(EventType.timeout)),
        (("shot", "attempt", "shoot"), lambda e: e.type in SHOT_TYPES),
    ]  # fmt: skip
    preds = [p for words, p in table if any(w in q for w in words)]
    for t in run.teams:  # team filter by name, color word, or "team a"
        names = [f"team {t.team_id.lower()}"] + ([t.name.lower()] if t.name else [])
        if any(n in q for n in names):
            preds.append(lambda e, tid=t.team_id: e.team_id == tid)
    return preds


def matching_events(run: AnalysisRun, question: str) -> list[Event]:
    preds = _filters(run, question)
    return [e for e in live_events(run) if all(p(e) for p in preds)]


def answer_stats(run: AnalysisRun, question: str) -> tuple[str, list[dict]]:
    events = live_events(run)
    shots = [e for e in events if e.type in SHOT_TYPES]
    made = [e for e in shots if shot_result(e) == "made"]
    missed = [e for e in shots if shot_result(e) == "missed"]
    other = Counter(e.type.value.replace("_", " ") for e in events if e.type not in SHOT_TYPES)
    lines = []
    for ts in run.team_summaries:
        s = ts.stats
        parts = [f"{s.pts} pts", f"FG {s.fgm}/{s.fga}" + (f" ({100 * s.fgm / s.fga:.0f}%)" if s.fga else "")]
        if s.tpa:
            parts.append(f"3PT {s.tpm}/{s.tpa}")
        if s.fta:
            parts.append(f"FT {s.ftm}/{s.fta}")
        parts += [f"{getattr(s, k)} {label}" for k, label in (("reb", "reb"), ("ast", "ast"), ("stl", "stl"), ("blk", "blk"), ("tov", "TO")) if getattr(s, k)]  # fmt: skip
        lines.append(f"- **{team_name(run, ts.team.team_id)}**: " + ", ".join(parts))
    if not shots and not lines and not other:
        return "The event log for this clip is empty, so there are no stats to report yet.", []
    unclear = len(shots) - len(made) - len(missed)
    text = f"From the event log: {len(shots)} shot attempts, {len(made)} made, {len(missed)} missed"
    text += f", {unclear} unclear." if unclear else "."
    if lines:
        text += "\n" + "\n".join(lines)
    if other:
        text += (
            "\nOther events: "
            + ", ".join(f"{n} {k}{'s' if n != 1 else ''}" for k, n in sorted(other.items()))
            + "."
        )
    pending = sum(1 for e in shots if e.status in ("provisional", "needs_review"))
    if pending:
        text += f"\n\n{pending} of these shots are not confirmed yet (provisional or waiting for review)."
    return text, [cite(run, e) for e in sorted(made + missed, key=event_time)][:12]


def answer_moments(run: AnalysisRun, question: str) -> tuple[str, list[dict]]:
    found = matching_events(run, question)
    if not found:
        kinds = sorted({event_label(run, e).split(",")[0].split(" (")[0] for e in live_events(run)})
        have = f" The log does have: {', '.join(kinds)}." if kinds else " The event log is empty."
        return "I couldn't find any matching moments in this clip's event log." + have, []
    shown = found[:20]
    lines = [f"- {clock(event_time(e))}: {event_label(run, e).rsplit(',', 1)[0]}" + ("" if e.status == "confirmed" else f" ({e.status.replace('_', ' ')})") for e in shown]  # fmt: skip
    more = f"\n...and {len(found) - len(shown)} more." if len(found) > len(shown) else ""
    noun = "moment" if len(found) == 1 else "moments"
    return f"Found {len(found)} {noun}:\n" + "\n".join(lines) + more, [cite(run, e) for e in shown]


# ---------------------------------------------------------------------------- VLM answers


def _target(run: AnalysisRun, question: str) -> tuple[float | None, Event | None]:
    events = live_events(run)
    for e in run.events:
        if re.search(rf"(?<![\w-]){re.escape(e.event_id)}(?![\w-])", question):
            return event_time(e), e
    t = parse_time(question)
    if t is not None:
        near = [e for e in events if abs(event_time(e) - t) <= 3.0]
        return t, min(near, key=lambda e: abs(event_time(e) - t)) if near else None
    cands = matching_events(run, question) if _filters(run, question) else []
    q = question.lower()
    if cands and any(w in q for w in ("last", "final", "latest")):
        return event_time(cands[-1]), cands[-1]
    if cands and any(w in q for w in ("first", "opening")):
        return event_time(cands[0]), cands[0]
    if len(cands) == 1:
        return event_time(cands[0]), cands[0]
    return None, None


def _tile_citations(text: str, index: dict[str, float]) -> list[dict]:
    out, seen = [], set()
    for m in TILE_RE.finditer(text or ""):
        tile = m.group(1).upper()
        t = resolve_ref(tile, index)
        if t is not None and tile not in seen:
            seen.add(tile)
            out.append({"t_s": t, "event_id": None, "label": f"Frame {tile}, {clock(t)}"})
    return out


def answer_explain(
    run: AnalysisRun,
    question: str,
    vlm: VisionLLM | None,
    video: Path | None,
    history: Sequence[tuple[str, str]],
) -> QAAnswer:
    t, ev = _target(run, question)
    duration = run.clip.master.duration_s
    base_cites = [cite(run, ev)] if ev else []
    if vlm is None or video is None:
        why = "No AI provider is selected" if vlm is None else "The video isn't available"
        if ev is not None:
            text = f"{why}, so here is what the event log says: {event_label(run, ev)}"
            text += f" ({ev.status.replace('_', ' ')}, confidence {ev.confidence:.0%})." + (
                f" {ev.description}" if ev.description else ""
            )
        else:
            text = (
                f"{why}, so I can't look at the play itself. Try asking about stats or when things happened."
            )
        return QAAnswer(text, "explain_play", base_cites)
    if t is not None:
        lo = (ev.t_start_s if ev else t) - 1.5
        hi = ((ev.t_end_s or event_time(ev)) if ev else t) + 1.5
        t0, t1 = max(0.0, lo), min(duration, hi) if duration else hi
        images, index = window_images(video, t0, t1, fps=6.0, sheet_prefix="W")
        nearby = [e for e in live_events(run) if t0 - 3 <= event_time(e) <= t1 + 3]
        focus = f"Focus on {clock(t0)} to {clock(t1)} of the clip; the sheets cover exactly that window."
        cache_key = f"{run.run_id}:qa:{t0:.1f}-{t1:.1f}"
    else:
        images, index = overview_images(video, duration, fps=1.0, max_sheets=4)
        nearby, focus, cache_key = None, "The sheets cover the whole clip.", f"{run.run_id}:qa:overview"
    res = vlm.analyze(
        AnalyzeRequest(
            prompt=load_prompt("qa", question=question, focus=focus),
            system=system_prompt(),
            images=images[: vlm.caps.max_images],
            cv_context=run_context(run, nearby),
            history=list(history),
            effort="medium",
            max_tokens=8000,
            purpose="qa",
            cache_key=cache_key,
        )
    )
    if res.error and not res.text:
        text = f"Sorry, the AI couldn't answer that ({res.error})."
    elif res.refused:
        text = "The AI declined to answer that question."
    else:
        text = res.text.strip()
    return QAAnswer(
        text,
        "explain_play",
        base_cites + _tile_citations(res.text, index),
        res.cost_usd or 0.0,
        res.provider,
        res.model_id,
    )


def answer_summary(run: AnalysisRun, vlm: VisionLLM | None, video: Path | None) -> QAAnswer:
    saved = [n for n in run.narratives if n.kind in ("recap", "play_by_play", "coaching", "custom")]
    if saved:
        n = saved[-1]
        by_id = {e.event_id: e for e in run.events}
        cites = [cite(run, by_id[i]) for i in n.cited_event_ids if i in by_id]
        cites += [{"t_s": t, "event_id": None, "label": clock(t)} for t in n.cited_times_s]
        provider, _, model = n.provenance.producer.partition(":")
        return QAAnswer(n.text, "summary", cites, 0.0, provider or None, model or None)
    if vlm is not None and video is not None:
        from hoops.reason.summarize import summarize

        summary, text, res = summarize(
            vlm,
            video=video,
            duration_s=run.clip.master.duration_s,
            preset="recap",
            cv_context=run_context(run),
            cache_key=f"{run.run_id}:summary",
        )
        cites = []
        for m in summary.key_moments if summary else []:
            t = resolve_ref(m.ref, {})
            if t is not None:
                cites.append({"t_s": t, "event_id": None, "label": f"{m.description[:60]}, {clock(t)}"})
        if not text:
            text = f"Sorry, the AI couldn't summarize the clip ({res.error or 'no answer'})."
        return QAAnswer(text, "summary", cites, res.cost_usd or 0.0, res.provider, res.model_id)
    stats, cites = answer_stats(run, "")
    return QAAnswer(
        "No AI summary is saved for this run yet. Here's what the event log says:\n\n" + stats,
        "summary",
        cites,
    )


# ---------------------------------------------------------------------------- entry point


def answer(
    run: AnalysisRun,
    question: str,
    *,
    vlm: VisionLLM | None,
    decider: Any = None,
    history: Sequence[tuple[str, str]] = (),
    video: Path | None = None,
) -> QAAnswer:
    """Answer one question about ``run`` (see the module docstring for routing)."""
    start = time.perf_counter()
    q = (question or "").strip()
    if not q:
        return QAAnswer(
            'Ask me something about this clip, for example "how many shots went in?"', "unsupported"
        )
    route, decision = route_question(q, decider, history)
    if route == "stats":
        text, cites = answer_stats(run, q)
        out = QAAnswer(text, "stats", cites)
    elif route == "find_moments":
        text, cites = answer_moments(run, q)
        out = QAAnswer(text, "find_moments", cites)
    elif route == "explain_play":
        out = answer_explain(run, q, vlm, video, history)
    elif route == "summary":
        out = answer_summary(run, vlm, video)
    else:
        out = QAAnswer(UNSUPPORTED_ANSWER, "unsupported")
    out.decision = decision
    out.cost_usd = round((out.cost_usd or 0.0) + ((decision or {}).get("cost_usd") or 0.0), 6)
    out.latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return out
