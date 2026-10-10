from __future__ import annotations

import json

import pytest

from hoops import registry
from hoops.decide.base import PRESETS, Answer, Choice, Noul, Score, gate, question_spec
from hoops.schemas import (
    AnalysisRun,
    ClipMeta,
    ClipSummary,
    Event,
    EventType,
    NormalizationReport,
    Provenance,
    VideoAsset,
    VLMResult,
    WindowVerdict,
)
from hoops.store import Store


def _asset(path="clips/x/master.mp4"):
    return VideoAsset(
        path=path, width=1280, height=720, fps="30000/1001", fps_float=29.97, n_frames=60, duration_s=2.0
    )


def _clip():
    return ClipMeta(
        clip_id="abc123",
        source_size_bytes=10,
        master=_asset(),
        proxy=_asset("clips/x/proxy.mp4"),
        normalization=NormalizationReport(summary="ok"),
    )


def test_run_roundtrip(data_dir):
    store = Store()
    run = AnalysisRun(run_id=Store.new_run_id(), clip=_clip())
    run.events.append(
        Event(
            event_id="e1",
            type=EventType.made_shot,
            t_start_s=1.0,
            confidence=0.9,
            confidence_kind="heuristic",
            provenance=Provenance(source="rule", producer="t", stage="shots"),
        )
    )
    store.save_run(run)
    loaded = store.load_run(run.run_id)
    assert loaded == run
    assert [r.run_id for r in store.list_runs()] == [run.run_id]


def _walk(schema, defs):
    """Yield every object schema (resolving $defs) in a JSON schema."""
    stack = [schema]
    seen = set()
    while stack:
        s = stack.pop()
        if not isinstance(s, dict):
            continue
        if "$ref" in s:
            name = s["$ref"].split("/")[-1]
            if name not in seen:
                seen.add(name)
                stack.append(defs[name])
            continue
        if s.get("type") == "object" and "properties" in s:
            yield s
        for v in s.values():
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


@pytest.mark.parametrize("model", [VLMResult, ClipSummary, WindowVerdict])
def test_vlm_models_are_strict_schema_friendly(model):
    """Anthropic + OpenAI strict subsets: all fields required, no extra, no numeric/length bounds."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    for obj in _walk(schema, defs):
        assert obj.get("additionalProperties") is False, obj.get("title")
        assert set(obj.get("required", [])) == set(obj["properties"]), obj.get("title")
    text = json.dumps(schema)
    for banned in ("minimum", "maximum", "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum"):
        assert banned not in text, banned


def test_gate_semantics():
    noul = Answer(kind="noul", value=0.9)
    assert gate(noul) == "accept"
    assert gate(Answer(kind="noul", value=0.1)) == "reject"
    assert gate(Answer(kind="noul", value=0.5)) == "escalate"
    assert gate(Answer(kind="choice", value="made", confidence=0.95)) == "accept"
    assert gate(Answer(kind="choice", value="made", confidence=0.5)) == "escalate"
    assert gate(Answer(kind="choice", value="made", confidence=None)) == "escalate"
    s = PRESETS["strict"]
    assert gate(Answer(kind="noul", value=0.9), s.accept, s.reject) == "escalate"


def test_question_spec():
    assert question_spec(Noul("x")) == {"kind": "noul", "instructions": "x"}
    assert question_spec(Choice("c", {"a": None}))["kind"] == "choice"
    assert question_spec(Score("s", ["lo", "hi"]))["criteria"] == ["lo", "hi"]


def test_registry_availability(monkeypatch):
    reg = registry.Registry("thing")

    @reg.register("needs-missing-module", label="x", requires_modules=("definitely_not_a_module_xyz",))
    def _a():
        return 1

    @reg.register("ok", label="y")
    def _b():
        return 2

    avail = reg.available()
    assert avail["ok"] is None and "definitely_not_a_module_xyz" in avail["needs-missing-module"]
    assert reg.create("ok") == 2
    with pytest.raises(RuntimeError):
        reg.create("needs-missing-module")
    with pytest.raises(KeyError):
        reg.create("nope")
