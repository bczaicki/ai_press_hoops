"""Video output that browsers can play, plus small ffprobe/ffmpeg helpers.

Never serve supervision.VideoSink / OpenCV VideoWriter output: their default 'mp4v' codec does not play
in Chrome/Safari. Everything user-facing goes through VideoWriter (H.264, yuv420p, +faststart).
"""

from __future__ import annotations

import json
import struct
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Self

import av
import av.logging
import numpy as np

from hoops.config import get_settings

av.logging.set_level(av.logging.ERROR)  # keep libx264's per-file statistics out of the console


class VideoWriter:
    """H.264 / yuv420p / +faststart writer (PyAV, bundled libx264). Frames are RGB uint8 (H, W, 3)."""

    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: str | float | Fraction,
        crf: int = 21,
        preset: str = "veryfast",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # yuv420p needs even dimensions; frames are cropped by one pixel if needed.
        self.width, self.height = width - width % 2, height - height % 2
        rate = Fraction(fps).limit_denominator(1001) if not isinstance(fps, Fraction) else fps
        self._container = av.open(str(self.path), "w", container_options={"movflags": "+faststart"})
        self._stream = self._container.add_stream("libx264", rate=rate)
        self._stream.width, self._stream.height = self.width, self.height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(crf), "preset": preset}
        self._closed = False

    def write(self, frame_rgb: np.ndarray) -> None:
        frame_rgb = np.ascontiguousarray(frame_rgb[: self.height, : self.width])
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        for packet in self._stream.encode(None):  # flush
            self._container.mux(packet)
        self._container.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def ffprobe_json(path: Path | str) -> dict:
    s = get_settings()
    out = subprocess.run(
        [s.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(out.stdout)


def mux_audio(video_path: Path, audio_src: Path, out_path: Path) -> Path:
    """Copy ``audio_src``'s first audio stream (if any) next to ``video_path``'s video, without re-encoding."""
    s = get_settings()
    cmd = [
        s.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, timeout=600)
    return out_path


def moov_before_mdat(path: Path | str) -> bool:
    """True when the MP4 index (moov) precedes the media data (mdat): playback can start immediately."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return False
            size, kind = struct.unpack(">I4s", header)
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:  # 64-bit size
                size = struct.unpack(">Q", f.read(8))[0]
                f.seek(size - 16, 1)
            elif size == 0:
                return False
            else:
                f.seek(size - 8, 1)


def is_browser_playable(path: Path | str) -> bool:
    info = ffprobe_json(path)
    video = next((st for st in info.get("streams", []) if st.get("codec_type") == "video"), None)
    return bool(
        video
        and video.get("codec_name") == "h264"
        and video.get("pix_fmt") == "yuv420p"
        and moov_before_mdat(path)
    )
