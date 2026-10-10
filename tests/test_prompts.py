from __future__ import annotations

import pytest

from hoops.reason.prompts import list_presets, load_prompt, system_prompt


def test_presets_match_api_contract_order():
    presets = list_presets()
    assert [p["key"] for p in presets] == ["recap", "play_by_play", "coach", "custom"]
    assert presets[2]["label"] == "Coach's notes" and all(p["description"] for p in presets)


def test_load_prompt_substitutes_and_keeps_json_braces():
    text = load_prompt("qa", question="Why was {that} a miss?", focus=None)
    assert "Why was {that} a miss?" in text and "{{" not in text and "\n\n\n" not in text
    assert load_prompt("custom", instructions="Count the dunks.") == "Count the dunks."
    assert "00:12.40" in load_prompt("adjudicate", t0="00:12.40", t1="00:15.90")


def test_system_prompt_has_shared_rules():
    s = system_prompt()
    for phrase in ("tile id", "ground truth", "Never name", '"unclear"'):
        assert phrase in s


@pytest.mark.parametrize("bad", ["../secrets", "nope", "Recap"])
def test_unknown_prompt_names_are_rejected(bad):
    with pytest.raises(KeyError):
        load_prompt(bad)
