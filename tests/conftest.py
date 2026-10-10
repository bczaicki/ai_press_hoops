"""Shared test fixtures. Videos are synthesized with ffmpeg lavfi so tests never need real footage."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hoops import config


def ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_ok(), reason="ffmpeg/ffprobe not on PATH")
live = pytest.mark.skipif(os.environ.get("HOOPS_LIVE") != "1", reason="set HOOPS_LIVE=1 to call paid APIs")


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Point settings.data_dir at a temp dir for the duration of a test."""
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("HOOPS_DATA_DIR", str(d))
    config.get_settings.cache_clear()
    yield d
    config.get_settings.cache_clear()


def make_video(
    path: Path,
    *,
    duration: float = 2.0,
    size: str = "640x360",
    fps: str = "30",
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    audio: bool = True,
    extra_out: list[str] | None = None,
    source: str | None = None,
) -> Path:
    """Write a synthetic test clip (testsrc2 pattern + optional sine audio)."""
    src = source or f"testsrc2=size={size}:rate={fps}:duration={duration}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-c:v", vcodec, "-pix_fmt", pix_fmt]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += list(extra_out or []) + [str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def testsrc_video(tmp_path) -> Path:
    return make_video(tmp_path / "testsrc.mp4")
