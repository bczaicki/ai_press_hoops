"""Shared fixtures. Videos are synthesized with ffmpeg lavfi so tests never need real footage."""

from __future__ import annotations

from pathlib import Path

import pytest

from hoops import config
from tests.helpers import make_video


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Point settings.data_dir at a temp dir for the duration of a test."""
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("HOOPS_DATA_DIR", str(d))
    config.get_settings.cache_clear()
    yield d
    config.get_settings.cache_clear()


@pytest.fixture
def testsrc_video(tmp_path) -> Path:
    return make_video(tmp_path / "testsrc.mp4")
