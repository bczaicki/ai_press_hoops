from __future__ import annotations

import io

import pytest
from PIL import Image

from hoops import cost
from hoops.reason.base import AnalyzeRequest, ImagePart, Usage
from hoops.schemas import ClipSummary


def _jpeg(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_image_tokens_by_provider():
    assert cost.image_tokens("anthropic", 1280, 720) == 46 * 26 == 1196
    assert cost.image_tokens("claude", 1920, 1080) == 69 * 39
    assert cost.image_tokens("openai", 1280, 720) == round(40 * 23 * 1.2)
    assert cost.image_tokens("openai", 10_000, 10_000) == 3000  # 2,500-patch cap
    assert cost.image_tokens("gemini", 1920, 1080) == 258
    assert cost.image_tokens("ollama", 1920, 1080) == 0


def test_video_and_text_tokens():
    assert cost.video_tokens_gemini(60) == 60 * 70 + 1500
    assert cost.video_tokens_gemini(10, fps=8, high_res=True, audio=False) == 80 * 280
    assert cost.text_tokens("abcd" * 10) == 10
    assert cost.text_tokens("") == 0


def test_price_lookup_and_base_model():
    assert cost.base_model("google/gemini-3.8-flash:free") == "gemini-3.8-flash"
    assert cost.price_for("anthropic/claude-opus-5-5") == (4.0, 20.0)
    assert cost.price_for("qwen3.5:4b") is None


def test_cost_from_usage_with_cache_and_tiers():
    u = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert cost.cost_from_usage("claude-opus-5-5", u) == pytest.approx(4.0 + 2.0)
    cached = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cost.cost_from_usage("claude-opus-5-5", cached) == pytest.approx(4.0 * 0.05 + 4.0 * 1.25)
    # Haiku 5.5: prompts above 100k tokens cost 5x
    assert cost.cost_from_usage("claude-haiku-5-5", Usage(200_000, 0)) == pytest.approx(0.2 * 0.5)
    assert cost.cost_from_usage("unknown-model", u) is None


def test_estimate_scales_with_images_and_is_zero_locally():
    small = AnalyzeRequest(prompt="Summarize the clip.", max_tokens=16000)
    big = AnalyzeRequest(
        prompt="Summarize the clip.",
        images=[ImagePart(jpeg=_jpeg(1920, 1080), label=f"S{i}") for i in range(6)],
        output_model=ClipSummary,
        max_tokens=16000,
    )
    e_small = cost.estimate(small, "claude", "claude-opus-5-5")
    e_big = cost.estimate(big, "claude", "claude-opus-5-5")
    assert 0 < e_small < e_big < 0.5
    # 6 sheets x 2,691 tokens at $4/M is ~ $0.065 of input
    assert e_big - e_small > 0.06
    assert cost.estimate(big, "ollama", "qwen3.5:4b") == 0.0
    assert cost.estimate(big, "openrouter", "google/gemini-3.8-flash") < e_big


def test_cost_guard_caps(monkeypatch):
    guard = cost.CostGuard(run_cap=0.10, session_cap=0.15)
    guard.check(0.05)
    guard.add(0.08)
    assert guard.spent == pytest.approx(0.08)
    with pytest.raises(cost.CostCapExceeded, match="run"):
        guard.check(0.05)
    guard.new_run()
    assert guard.spent == 0.0 and guard.session_spent == pytest.approx(0.08)
    with pytest.raises(cost.CostCapExceeded, match="session"):
        guard.check(0.09)
    guard.check(0.05)


def test_cost_guard_defaults_from_settings(data_dir):
    guard = cost.CostGuard()
    assert guard.run_cap == 0.50 and guard.session_cap == 5.00
