"""Score runs against hand labels: ±1 s matching, P/R/F1 with Wilson CIs, threshold sweeps, AUROC.

Label file (eval/labels/<name>.json, see eval/README.md)::

    {"clip_id": "3f9c...", "footage_type": "fixed_camera",
     "events": [{"t_s": 12.4, "type": "made_shot", "note": "corner 3"}, {"t_s": 31.0, "type": "missed_shot"}]}

Every labeled event is a shot attempt; ``made_shot``/``missed_shot`` also give the outcome, and
``shot_attempt`` means the outcome wasn't visible.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, computed_field

from hoops.schemas import AnalysisRun, Event, EventType

LABEL_TYPES = ("shot_attempt", "made_shot", "missed_shot")
SHOT_TYPES = frozenset(
    {
        EventType.shot_attempt,
        EventType.made_shot,
        EventType.missed_shot,
        EventType.three_pointer,
        EventType.free_throw,
    }
)


# ---------------------------------------------------------------- labels


def validate_labels(labels: dict[str, Any]) -> dict[str, Any]:
    """Check the label format; returns the labels with events sorted by time."""
    if not isinstance(labels.get("clip_id"), str):
        raise ValueError("labels need a 'clip_id' string")  # noqa: TRY004 - one error type for bad files
    events = labels.get("events")
    if not isinstance(events, list):
        raise ValueError("labels need an 'events' list")  # noqa: TRY004
    for i, e in enumerate(events):
        if not isinstance(e.get("t_s"), int | float) or e["t_s"] < 0:
            raise ValueError(f"event {i}: 't_s' must be seconds >= 0")
        if e.get("type") not in LABEL_TYPES:
            raise ValueError(f"event {i}: 'type' must be one of {LABEL_TYPES}")
    return {**labels, "events": sorted(events, key=lambda e: e["t_s"])}


def load_labels(path: Path) -> dict[str, Any]:
    return validate_labels(json.loads(Path(path).read_text()))


def labels_from_run(run: AnalysisRun) -> dict[str, Any]:
    """Gold labels from a reviewed run: shot events a human accepted or added (for eval/labels/)."""
    out = []
    for shot in predicted_shots(
        e for e in run.events if e.review in ("accepted", "edited") or e.provenance.source == "human"
    ):
        kind = "shot_attempt" if shot.made is None else ("made_shot" if shot.made else "missed_shot")
        out.append({"t_s": round(shot.t_s, 2), "type": kind})
    return {"clip_id": run.clip.clip_id, "footage_type": run.clip.footage_type, "events": out}


# ---------------------------------------------------------------- predictions


@dataclass
class PredShot:
    """One predicted shot: linked shot-family events (attempt + its make/miss) collapsed together."""

    t_s: float
    made: bool | None
    confidence: float
    event_ids: list[str]
    features: dict[str, Any] = field(default_factory=dict)


def _t(e: Event) -> float:
    return e.t_key_s if e.t_key_s is not None else e.t_start_s


def predicted_shots(events: Iterable[Event], *, include_rejected: bool = False) -> list[PredShot]:
    shots = [
        e
        for e in events
        if e.type in SHOT_TYPES and (include_rejected or (e.status != "rejected" and e.review != "rejected"))
    ]
    by_id = {e.event_id: e for e in shots}
    parent = {eid: eid for eid in by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in shots:
        for rid in e.related_event_ids:
            if rid in by_id:
                parent[find(rid)] = find(e.event_id)
    groups: dict[str, list[Event]] = defaultdict(list)
    for e in shots:
        groups[find(e.event_id)].append(e)
    out = []
    for members in groups.values():
        made = any(m.type == EventType.made_shot or (m.shot and m.shot.result == "made") for m in members)
        missed = any(
            m.type == EventType.missed_shot or (m.shot and m.shot.result == "missed") for m in members
        )
        features: dict[str, Any] = {}
        for m in members:
            for k, v in m.features.items():
                if isinstance(v, int | float) and not isinstance(v, bool):
                    features[k] = max(v, features.get(k, v))
                else:
                    features.setdefault(k, v)
        out.append(
            PredShot(
                t_s=min(_t(m) for m in members),
                made=True if made else (False if missed else None),
                confidence=max(m.confidence for m in members),
                event_ids=[m.event_id for m in members],
                features=features,
            )
        )
    return sorted(out, key=lambda s: s.t_s)


# ---------------------------------------------------------------- metrics


def match(
    pred_times: Sequence[float], gold_times: Sequence[float], tol_s: float = 1.0
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one greedy matching, closest pairs first, within ``tol_s``.

    Returns (matches as (pred_index, gold_index), unmatched pred indices = false alarms,
    unmatched gold indices = misses).
    """
    pairs = sorted(
        (abs(p - g), i, j)
        for i, p in enumerate(pred_times)
        for j, g in enumerate(gold_times)
        if abs(p - g) <= tol_s + 1e-9
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j))
    matches.sort()
    fp = [i for i in range(len(pred_times)) if i not in used_p]
    fn = [j for j in range(len(gold_times)) if j not in used_g]
    return matches, fp, fn


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (n=0 -> (0, 1): we know nothing)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def auroc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Probability a random correct call outscores a random wrong one (ties count half). NaN if one class."""
    pos = [s for s, c in zip(scores, correct, strict=True) if c]
    neg = [s for s, c in zip(scores, correct, strict=True) if not c]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class PRF(BaseModel):
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> PRF:
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
        return cls(
            tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1,
            precision_ci=wilson(tp, tp + fp), recall_ci=wilson(tp, tp + fn),
        )  # fmt: skip


class EvalReport(BaseModel):
    clip_id: str | None
    run_id: str | None
    footage_type: str
    tol_s: float
    n_clips: int = 1
    attempts: PRF
    makes: PRF
    outcome_correct: int = 0  # matched attempts whose made/missed call agrees with the label
    outcome_n: int = 0  # matched attempts where both sides know the outcome
    confidence_auroc: float | None = None  # does confidence separate right from wrong attempt calls?
    false_alarms_s: list[float] = []
    missed_s: list[float] = []

    @computed_field
    @property
    def outcome_accuracy(self) -> float | None:
        return self.outcome_correct / self.outcome_n if self.outcome_n else None


def score_run(run: AnalysisRun, labels: dict[str, Any], tol_s: float = 1.0) -> EvalReport:
    labels = validate_labels(labels)
    gold = labels["events"]
    gold_t = [e["t_s"] for e in gold]
    shots = predicted_shots(run.events)
    pred_t = [s.t_s for s in shots]
    matches, fp, fn = match(pred_t, gold_t, tol_s)
    attempts = PRF.from_counts(len(matches), len(fp), len(fn))

    made_pred = [s.t_s for s in shots if s.made]
    made_gold = [e["t_s"] for e in gold if e["type"] == "made_shot"]
    m_matches, m_fp, m_fn = match(made_pred, made_gold, tol_s)

    outcome_n = outcome_ok = 0
    for i, j in matches:
        truth = {"made_shot": True, "missed_shot": False}.get(gold[j]["type"])
        if truth is not None and shots[i].made is not None:
            outcome_n += 1
            outcome_ok += shots[i].made == truth
    correct = [i in {m[0] for m in matches} for i in range(len(shots))]
    roc = auroc([s.confidence for s in shots], correct)
    return EvalReport(
        clip_id=labels["clip_id"],
        run_id=run.run_id,
        footage_type=labels.get("footage_type") or run.clip.footage_type,
        tol_s=tol_s,
        attempts=attempts,
        makes=PRF.from_counts(len(m_matches), len(m_fp), len(m_fn)),
        outcome_correct=outcome_ok,
        outcome_n=outcome_n,
        confidence_auroc=None if math.isnan(roc) else roc,
        false_alarms_s=[round(pred_t[i], 2) for i in fp],
        missed_s=[round(gold_t[j], 2) for j in fn],
    )


def combine(reports: Sequence[EvalReport], footage_type: str | None = None) -> EvalReport:
    """Pool counts across clips (AUROC isn't poolable from summaries, so it's dropped)."""
    if not reports:
        raise ValueError("nothing to combine")

    def pool(name: str) -> PRF:
        prfs = [getattr(r, name) for r in reports]
        return PRF.from_counts(sum(p.tp for p in prfs), sum(p.fp for p in prfs), sum(p.fn for p in prfs))

    types = {r.footage_type for r in reports}
    return EvalReport(
        clip_id=None,
        run_id=None,
        footage_type=footage_type or (types.pop() if len(types) == 1 else "mixed"),
        tol_s=reports[0].tol_s,
        n_clips=sum(r.n_clips for r in reports),
        attempts=pool("attempts"),
        makes=pool("makes"),
        outcome_correct=sum(r.outcome_correct for r in reports),
        outcome_n=sum(r.outcome_n for r in reports),
    )


def by_footage_type(reports: Sequence[EvalReport]) -> dict[str, EvalReport]:
    groups: dict[str, list[EvalReport]] = defaultdict(list)
    for r in reports:
        groups[r.footage_type].append(r)
    return {k: combine(v, k) for k, v in sorted(groups.items())}


def sweep(
    events: Sequence[Event],
    labels: dict[str, Any],
    *,
    score_key: str = "make_score",
    thresholds: Sequence[float] | None = None,
    target: Literal["make", "attempt"] | None = None,
    tol_s: float = 1.0,
) -> list[tuple[float, float, float]]:
    """(threshold, precision, recall) when keeping shots with features[score_key] >= threshold.

    Re-derives calls from stored scores (rejected events included: the threshold is what's being
    varied). ``target`` defaults to "make" when the key mentions make, else "attempt". Precision is
    1.0 when nothing passes (the PR-curve convention).
    """
    labels = validate_labels(labels)
    target = target or ("make" if "make" in score_key else "attempt")
    gold = [e["t_s"] for e in labels["events"] if target == "attempt" or e["type"] == "made_shot"]
    scored = [
        (s.t_s, float(s.features[score_key]))
        for s in predicted_shots(events, include_rejected=True)
        if isinstance(s.features.get(score_key), int | float)
    ]
    if thresholds is None:
        thresholds = [round(i / 20, 2) for i in range(21)]
    out = []
    for thr in thresholds:
        kept = [t for t, score in scored if score >= thr]
        matches, _, _ = match(kept, gold, tol_s)
        precision = len(matches) / len(kept) if kept else 1.0
        recall = len(matches) / len(gold) if gold else 1.0
        out.append((float(thr), precision, recall))
    return out


# ---------------------------------------------------------------- reporting


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def report_markdown(report: EvalReport | Sequence[EvalReport]) -> str:
    """Markdown for the UI / CLI. A list renders one section per footage type plus an overall row."""
    if not isinstance(report, EvalReport):
        reports = list(report)
        if not reports:
            return "No labeled clips to score yet. See eval/README.md."
        sections = [report_markdown(r) for r in by_footage_type(reports).values()]
        if len({r.footage_type for r in reports}) > 1:
            sections.append(report_markdown(combine(reports, "all footage")))
        return "\n\n".join(sections)
    r = report
    who = f"`{r.clip_id}`" if r.clip_id else f"{r.n_clips} clip(s)"
    lines = [
        f"### {r.footage_type}: {who}, matched within ±{r.tol_s:g} s",
        "",
        "| | Precision (95% CI) | Recall (95% CI) | F1 | Right | False alarms | Missed |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, prf in (("Shot attempts", r.attempts), ("Made shots", r.makes)):
        p_ci, r_ci = prf.precision_ci, prf.recall_ci
        lines.append(
            f"| {label} | {_pct(prf.precision)} ({p_ci[0]:.0%}–{p_ci[1]:.0%}) "
            f"| {_pct(prf.recall)} ({r_ci[0]:.0%}–{r_ci[1]:.0%}) | {_pct(prf.f1)} "
            f"| {prf.tp} | {prf.fp} | {prf.fn} |"
        )
    lines.append("")
    if r.outcome_n:
        lines.append(
            f"- Made/missed calls on found shots: {_pct(r.outcome_accuracy)} right (n={r.outcome_n})"
        )
    if r.confidence_auroc is not None:
        lines.append(
            f"- Confidence AUROC {r.confidence_auroc:.2f} (1.0 = confidence perfectly flags mistakes, "
            "0.5 = it tells you nothing)"
        )
    if r.false_alarms_s:
        lines.append("- False alarms at " + ", ".join(_mmss(t) for t in r.false_alarms_s))
    if r.missed_s:
        lines.append("- Missed shots at " + ", ".join(_mmss(t) for t in r.missed_s))
    if r.attempts.tp + r.attempts.fn < 30:
        lines.append("- Fewer than 30 labeled shots: treat these numbers as rough (see the CI widths).")
    return "\n".join(lines).rstrip()


def run_eval(labels_dir: Path, store: Any = None, tol_s: float = 1.0) -> tuple[list[EvalReport], list[str]]:
    """Score the newest finished run of each labeled clip. Returns (reports, notes about skipped files)."""
    from hoops.store import Store

    store = store or Store()
    runs = [s for s in store.list_runs() if s.status == "done"]  # newest first
    reports, notes = [], []
    for path in sorted(Path(labels_dir).glob("*.json")):
        try:
            labels = load_labels(path)
        except (ValueError, json.JSONDecodeError) as exc:
            notes.append(f"{path.name}: bad label file ({exc})")
            continue
        summary = next((s for s in runs if s.clip_id == labels["clip_id"]), None)
        if summary is None:
            notes.append(f"{path.name}: no finished run for clip {labels['clip_id']}")
            continue
        reports.append(score_run(store.load_run(summary.run_id), labels, tol_s))
    return reports, notes
