"""System One question sets and small state builders.

Instructions are short and literal. States hold only short categorical text written by code: Jev
can't count, compare numbers, or do arithmetic, and it gets less accurate when the state has
irrelevant detail. The one exception is ``event_state(..., with_scores=True)``, which only the
rules baseline asks for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hoops.decide.base import Choice, Noul, Question

if TYPE_CHECKING:
    from hoops.events.shots import Candidate

TRIAGE: dict[str, Question] = {
    "footage_type": Choice(
        "What kind of basketball video is this?",
        {
            "broadcast": "TV or stream coverage: the camera follows play, with cuts, replays, a scoreboard",
            "fixed_camera": "one steady camera, usually on a tripod, showing a half or full court",
            "handheld": "a phone or camera held by a person; it moves and shakes",
            "highlights": "an edit of short highlight clips with many cuts",
            "practice": "drills, warm-ups, or shooting practice rather than a game",
            "not_basketball": "not basketball footage",
        },
    ),
    "court_lines_visible": Noul("The court's painted lines are clearly visible."),
}

EVENT_TYPING: dict[str, Question] = {
    "is_shot_attempt": Noul("A player shot the ball at the basket in this moment."),
    "outcome": Choice(
        "Did the shot go in?",
        {
            "made": "the ball went through the hoop",
            "missed": "the ball did not go through the hoop",
            "unclear": "the description does not show whether it went in",
        },
    ),
    "shot_kind": Choice(
        "What kind of shot was it?",
        {
            "jumper": "a jump shot from away from the rim",
            "layup": "a close shot laid in off the backboard or rim",
            "dunk": "the ball was pushed down through the rim",
            "free_throw": "an uncontested shot from the free-throw line",
            "tip": "a quick tap of a missed shot",
            "unknown": "the description does not show the kind of shot",
        },
    ),
}

QA_ROUTE: dict[str, Question] = {
    "route": Choice(
        "Which tool should answer this question about the game video?",
        {
            "stats": "counts, totals, or percentages, such as points, shots, or makes",
            "find_moments": "find or list moments in the video: when, where, show me",
            "explain_play": "explain how or why a specific play happened",
            "summary": "a general recap or summary of the video",
            "unsupported": "anything else, or not about this video",
        },
    ),
}

CLAIM_CHECK: dict[str, Question] = {
    "supported": Noul("The event log supports this claim."),
}

QUESTION_SETS: dict[str, dict[str, Question]] = {
    "triage": TRIAGE,
    "event_typing": EVENT_TYPING,
    "qa_route": QA_ROUTE,
    "claim_check": CLAIM_CHECK,
}


def _short(value: Any, limit: int = 300) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def event_state(c: Candidate, *, with_scores: bool = False) -> dict[str, Any]:
    """Event-typing state: the candidate's descriptors (short categorical text).

    ``with_scores`` adds the numeric rule scores. Only the rules baseline uses them; they are never
    sent to Jev or an LLM.
    """
    state: dict[str, Any] = {
        "moment": "basketball video, described from ball and rim tracking",
        **c.descriptors,
    }
    if with_scores:
        state["rule_scores"] = {"attempt": round(c.attempt_score, 4), "make": round(c.make_score, 4)}
    return state


def triage_state(**facts: Any) -> dict[str, str]:
    """Footage-triage state from short facts. None and empty values are dropped; long text is cut.

    Suggested keys (all short words): resolution, frame_rate, duration, scene_cuts ("none" | "few" |
    "many"), camera_motion ("steady" | "panning" | "shaky"), rim ("one, steady" | "one, moving" |
    "several" | "not seen"), players ("many" | "few" | "none"), scoreboard ("visible" | "not seen"),
    court_lines ("visible" | "not visible"), caption (one line from a VLM).
    """
    return {k: _short(v) for k, v in facts.items() if v not in (None, "", [], {})}


def qa_state(question: str, digest: str) -> dict[str, str]:
    """Q&A routing state: the user's question plus a one-paragraph digest of what the run has."""
    return {"question": _short(question, 500), "video_data_available": _short(digest, 2000)}


def claim_state(sentence: str, events_digest: str) -> dict[str, str]:
    """Claim-check state: one narrative sentence plus the event log as short text."""
    return {"claim": _short(sentence, 500), "event_log": _short(events_digest, 6000)}
