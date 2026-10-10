from __future__ import annotations

from types import SimpleNamespace

from hoops.reason.openai_compat import OllamaVLM


class _Client:
    def __init__(self, names):
        self._names = names

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=n) for n in self._names])


def test_falls_back_to_installed_vision_model():
    vlm = OllamaVLM(client=_Client(["nomic-embed-text:latest", "gemma4:latest"]))
    assert vlm.resolve_model("qwen3.5:4b", explicit=False) == "gemma4:latest"


def test_keeps_installed_or_explicit_model():
    vlm = OllamaVLM(client=_Client(["qwen3.5:4b", "gemma4:latest"]))
    assert vlm.resolve_model("qwen3.5:4b", explicit=False) == "qwen3.5:4b"
    vlm2 = OllamaVLM(model="llava:13b", client=_Client(["gemma4:latest"]))
    assert vlm2.resolve_model("llava:13b", explicit=True) == "llava:13b"


def test_no_server_keeps_wanted():
    class Down:
        def list(self):
            raise ConnectionError("down")

    assert OllamaVLM(client=Down()).resolve_model("qwen3.5:4b", explicit=False) == "qwen3.5:4b"
