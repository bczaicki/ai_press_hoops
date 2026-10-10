from __future__ import annotations

from hoops import config, doctor
from hoops.doctor import Check, render_text, run_doctor

KEY_ENVS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "TYPESAFE_API_KEY",
    "ROBOFLOW_API_KEY",
)


def test_doctor_runs_without_keys(data_dir, monkeypatch):
    for k in KEY_ENVS:
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens on the discard port
    config.get_settings.cache_clear()
    checks = run_doctor()
    names = {c.name for c in checks}
    assert {"python", "ffmpeg", "ollama", "data dir", "key:ANTHROPIC_API_KEY"} <= names
    assert all(isinstance(c, Check) for c in checks)
    keys = [c for c in checks if c.name.startswith("key:")]
    assert keys and not any(c.ok for c in keys) and all(c.severity == "warn" for c in keys)
    assert next(c for c in checks if c.name == "ollama").ok is False
    assert next(c for c in checks if c.name == "data dir").ok
    text = render_text(checks)
    assert "error(s)" in text and "ANTHROPIC_API_KEY" in text


def test_doctor_never_prints_key_values(data_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRET-VALUE-123")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    config.get_settings.cache_clear()
    checks = run_doctor()
    assert next(c for c in checks if c.name == "key:ANTHROPIC_API_KEY").ok
    assert "SECRET-VALUE" not in render_text(checks)


def test_a_crashing_check_is_reported_not_raised(data_dir, monkeypatch):
    def boom():
        raise RuntimeError("kaput")

    monkeypatch.setattr(doctor, "_torch_mps", boom)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    config.get_settings.cache_clear()
    crashed = next(c for c in run_doctor() if c.name == "torch MPS")
    assert not crashed.ok and "kaput" in crashed.detail
