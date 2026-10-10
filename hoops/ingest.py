"""T0 ingest: probe -> decide -> normalize -> cache.

Every upload becomes, once, in data/clips/<clip_id>/:
- master.mp4: upright, <=1080p (either orientation), square pixels, SDR BT.709, constant frame rate,
  H.264 High yuv420p + AAC, +faststart, metadata stripped (iPhone GPS). The playback/export copy.
- proxy_720p.mp4: <=720p, <=30 fps CFR, 1 s GOP, no audio, same t=0. What CV and samplers read.
- poster.jpg, probe.json (raw ffprobe of the source), clip.json (ClipMeta).

The decision table (`plan_normalization`) is a pure function of the ffprobe JSON and the ffmpeg build's
filters, so every branch is unit-testable without exotic sample files.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from hoops.config import get_settings
from hoops.schemas import ClipMeta, NormalizationReport, VideoAsset
from hoops.store import Store
from hoops.video_io import ffprobe_json

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

STANDARD_FPS = ("24000/1001", "24", "25", "30000/1001", "30", "50", "60000/1001", "60")
HDR_TRANSFERS = {"smpte2084": "pq", "arib-std-b67": "hlg"}
SD_MATRICES = {"smpte170m", "bt470bg", "fcc", "smpte240m"}  # converted to BT.709 instead of just re-tagged
INTERLACE_CODECS = {"mpeg2video", "mpeg1video", "h264", "hevc", "dvvideo", "mpeg4", "prores", "dnxhd", "vc1"}
IMAGE_FORMATS = {"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe", "heif"}
MIN_DURATION_S = 0.3

# Share of the progress bar spent in each step.
_P_MASTER, _P_PROXY = 0.75, 0.93


class IngestError(Exception):
    """A problem with the upload, phrased so it can be shown to the user as-is."""


# ---------------------------------------------------------------- ffmpeg capabilities


@dataclass(frozen=True)
class FFmpegCaps:
    version: str
    filters: frozenset[str]
    encoders: frozenset[str]


@functools.lru_cache(maxsize=4)
def ffmpeg_capabilities(ffmpeg_bin: str | None = None) -> FFmpegCaps:
    """Version, filters and encoders of the ffmpeg we call. Raises FileNotFoundError if it's missing."""
    exe = ffmpeg_bin or get_settings().ffmpeg_bin

    def run(arg: str) -> str:
        cmd = [exe, "-hide_banner", arg]
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30).stdout

    first = (run("-version").splitlines() or [""])[0]
    m = re.match(r"ffmpeg version (\S+)", first)
    filters = {p[1] for line in run("-filters").splitlines() if len(p := line.split()) >= 3 and "->" in p[2]}
    encoders = {
        p[1]
        for line in run("-encoders").splitlines()
        if len(p := line.split()) >= 2 and re.fullmatch(r"[VASFXBD.]{6}", p[0]) and p[1] != "="
    }
    return FFmpegCaps(m.group(1) if m else first, frozenset(filters), frozenset(encoders))


# ---------------------------------------------------------------- probe + decision table


def probe(path: Path) -> dict:
    """ffprobe JSON (format + streams). Raises IngestError for files ffprobe can't read."""
    try:
        return ffprobe_json(path)
    except FileNotFoundError as exc:
        raise IngestError("ffprobe isn't installed, so videos can't be read. Run `hoops doctor`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError("Reading this file took too long; it may be damaged.") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise IngestError("This file doesn't look like a video we can read.") from exc


@dataclass
class Plan:
    """What the master encode will do, derived from the probe."""

    filters: list[str]
    fps: str  # target rational, e.g. "30000/1001"
    duration_s: float
    has_audio: bool
    input_args: list[str] = field(default_factory=list)  # e.g. hardware device init for scale_vt
    report: NormalizationReport = field(default_factory=NormalizationReport)


def video_stream(info: dict) -> dict | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic"):
            return s
    return None


def _rate(value: str | None) -> Fraction | None:
    try:
        f = Fraction(value) if value else None
    except (ValueError, ZeroDivisionError):
        return None
    return f if f and f > 0 else None


def snap_fps(fps: float) -> str:
    """Nearest standard rate, capped at 60 (119.88/239.76 -> 59.94, 120/240 -> 60)."""
    if fps > 60.5:
        return "60000/1001" if abs(fps - round(fps)) > 0.05 else "60"
    return min(STANDARD_FPS, key=lambda r: abs(float(Fraction(r)) - fps))


def fps_label(rate: str | float) -> str:
    f = float(Fraction(rate)) if isinstance(rate, str) else rate
    return str(round(f)) if abs(f - round(f)) < 1e-3 else f"{f:.3f}".rstrip("0").rstrip(".")


def _side_data(stream: dict, kind: str) -> dict | None:
    want = kind.replace(" ", "").lower()
    for d in stream.get("side_data_list", []) or []:
        if str(d.get("side_data_type", "")).replace(" ", "").lower() == want:
            return d
    return None


def rotation_of(stream: dict) -> int:
    """Counter-clockwise display rotation in degrees, normalized to 0/90/180/270."""
    dm = _side_data(stream, "Display Matrix")
    if dm is not None and dm.get("rotation") is not None:
        deg = float(dm["rotation"])
    elif (tag := (stream.get("tags") or {}).get("rotate")) is not None:
        deg = -float(tag)  # legacy tag is clockwise
    else:
        return 0
    return round(deg / 90.0) * 90 % 360


def _sar(stream: dict) -> Fraction:
    sar = stream.get("sample_aspect_ratio") or "1:1"
    try:
        num, den = (int(x) for x in sar.split(":"))
        return Fraction(num, den) if num > 0 and den > 0 else Fraction(1)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)


def _fit(long_edge: int, short_edge: int, extra: str = "") -> str:
    """Orientation-aware downscale-only fit with even dimensions."""
    return (
        f"scale=w='if(gte(iw,ih),min(iw,{long_edge}),min(iw,{short_edge}))'"
        f":h='if(gte(iw,ih),min(ih,{short_edge}),min(ih,{long_edge}))'"
        f":force_original_aspect_ratio=decrease:force_divisible_by=2{extra}"
    )


def _duration(info: dict, vs: dict) -> float:
    for value in (info.get("format", {}).get("duration"), vs.get("duration")):
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except ValueError:
            continue
    return 0.0


def _tonemap_chain(
    vs: dict, transfer: str, filters: frozenset[str]
) -> tuple[list[str], list[str], str | None]:
    """(filters, input_args, method) to bring HDR to SDR BT.709 with what this ffmpeg build has."""
    if {"zscale", "tonemap"} <= filters:
        prim = vs.get("color_primaries") if vs.get("color_primaries") in {"bt2020", "bt709"} else "bt2020"
        matrix = vs.get("color_space") if vs.get("color_space") in {"bt2020nc", "bt2020c"} else "bt2020nc"
        return (
            [
                f"zscale=tin={transfer}:pin={prim}:min={matrix}:rin=tv:t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
            ],
            [],
            "zscale",
        )
    if (
        "scale_vt" in filters
    ):  # Apple VideoToolbox: no zimg needed (software decode, so autorotate still applies)
        return (
            [
                "format=p010le",
                "hwupload",
                "scale_vt=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709",
                "hwdownload",
                "format=p010le",  # scale_vt keeps the upload's sw format
            ],
            ["-init_hw_device", "videotoolbox=vt", "-filter_hw_device", "vt"],
            "scale_vt",
        )
    return [], [], None


def plan_normalization(info: dict, filters: Iterable[str], *, interlaced: bool | None = None) -> Plan:
    """The ingest decision table. ``interlaced`` is the idet verdict when field_order was unknown."""
    filters = frozenset(filters)
    vs = video_stream(info)
    if vs is None:
        raise IngestError("This file has no video track.")
    if info.get("format", {}).get("format_name") in IMAGE_FORMATS:
        raise IngestError("That's a photo, not a video. Please upload a video clip.")
    duration = _duration(info, vs)
    if duration < MIN_DURATION_S:
        raise IngestError("This video is too short (or its length can't be read).")
    max_s = get_settings().max_clip_seconds
    if duration > max_s:
        raise IngestError(
            f"This video is {duration / 60:.1f} minutes long; the limit is {max_s / 60:.0f} minutes. "
            "Please trim it and upload again."
        )

    report = NormalizationReport(source_resolution=None)
    chain: list[str] = []
    input_args: list[str] = []

    # Dolby Vision profile 5 has no backward-compatible base layer: colors would come out wrong.
    dovi = _side_data(vs, "DOVI configuration record")
    if dovi is not None and int(dovi.get("dv_profile", -1)) == 5:
        raise IngestError(
            "This video uses Dolby Vision profile 5 (common in streaming downloads), which can't be "
            "converted to normal colors here. Please export it as standard (SDR) or HDR10 video."
        )

    # Rotation: the ffmpeg CLI autorotates by default; we only record it (and size the output for it).
    report.rotation_deg = rotation_of(vs)
    w, h = int(vs.get("width") or 0), int(vs.get("height") or 0)
    sar = _sar(vs)
    disp_w, disp_h = round(w * sar), h
    if report.rotation_deg in (90, 270):
        disp_w, disp_h = disp_h, disp_w
    report.source_resolution = (disp_w, disp_h)

    # Interlacing: trust field_order when set, otherwise the idet verdict.
    field_order = vs.get("field_order")
    if field_order in {"tt", "bb", "tb", "bt"}:
        chain.append("bwdif=mode=send_frame:parity=auto:deint=interlaced")
        report.deinterlaced = True
    elif interlaced:
        chain.append("bwdif=mode=send_frame:parity=auto:deint=all")  # frames may not be flagged
        report.deinterlaced = True

    if sar != 1:
        chain += ["scale=trunc(iw*sar/2)*2:ih", "setsar=1"]
        report.sar_fixed = True

    # HDR: tone-map after downscaling (the float path is slow at 4K).
    transfer = vs.get("color_transfer")
    hdr = HDR_TRANSFERS.get(transfer or "")
    sd_matrix = not hdr and vs.get("color_space") in SD_MATRICES
    chain.append(_fit(1920, 1080, ":out_range=tv" + (":out_color_matrix=bt709" if sd_matrix else "")))
    if hdr:
        report.hdr = "dolby_vision" if dovi is not None else hdr  # type: ignore[assignment]
        tm, input_args, method = _tonemap_chain(vs, transfer or "", filters)
        chain += tm
        report.tonemapped = method is not None
        if dovi is not None:
            report.warnings.append("Dolby Vision extras were ignored; converted the standard HDR layer.")
        if method == "scale_vt":
            report.warnings.append(
                "HDR converted with Apple VideoToolbox because this ffmpeg lacks zscale; colors may differ "
                "slightly. `brew install ffmpeg-full` restores the default path."
            )
        elif method is None:
            report.warnings.append(
                "This ffmpeg can't convert HDR to normal colors (missing zscale/tonemap), so the video may "
                "look washed out. Run `hoops doctor` for the fix."
            )

    # Frame rate: snap to a standard rate (VFR phone video becomes exact frame indices).
    r, avg = _rate(vs.get("r_frame_rate")), _rate(vs.get("avg_frame_rate"))
    src = avg or r or Fraction(30)
    report.vfr_source = bool(r and avg and abs(r - avg) / r > 0.005)
    fps = snap_fps(float(src))
    if abs(float(Fraction(fps)) - float(src)) > 0.01:
        report.fps_snapped_from = fps_label(float(src))

    chain += [
        f"fps={fps}",
        "format=yuv420p",
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
    ]
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return Plan(chain, fps, duration, has_audio, input_args, report)


def needs_idet(info: dict) -> bool:
    vs = video_stream(info) or {}
    return vs.get("field_order") in (None, "unknown") and vs.get("codec_name") in INTERLACE_CODECS


def detect_interlaced(path: Path, frames: int = 300) -> bool:
    """Run ffmpeg's idet filter over the first frames; True when most frames look interlaced."""
    cmd = [get_settings().ffmpeg_bin, "-hide_banner", "-nostdin", "-protocol_whitelist", "file"]
    cmd += [
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        "idet",
        "-frames:v",
        str(frames),
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=180).stderr
    except (subprocess.TimeoutExpired, OSError):
        return False
    hits = re.findall(r"Multi frame detection: TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)", out)
    if not hits:
        return False
    tff, bff, prog = (int(x) for x in hits[-1])
    return tff + bff >= 10 and tff + bff > prog


def summarize(report: NormalizationReport, master: VideoAsset) -> str:
    parts = []
    if report.rotation_deg:
        parts.append(f"Rotated {min(report.rotation_deg, 360 - report.rotation_deg)}°")
    if report.tonemapped:
        parts.append("HDR→SDR")
    elif report.hdr != "none":
        parts.append("HDR (colors not converted)")
    if report.deinterlaced:
        parts.append("deinterlaced")
    if report.sar_fixed:
        parts.append("square pixels")
    rate = f"{fps_label(master.fps)} fps constant frame rate"
    if report.vfr_source:
        rate += " (was variable)"
    parts.append(rate)
    short = min(master.width, master.height)
    res = f"{short}p" + (" portrait" if master.height > master.width else "")
    if report.source_resolution and min(report.source_resolution) > short:
        res += f" (from {min(report.source_resolution)}p)"
    parts.append(res)
    if not master.has_audio:
        parts.append("no audio")
    text = ", ".join(parts)
    return text[0].upper() + text[1:]


# ---------------------------------------------------------------- ffmpeg runner


def _run_ffmpeg(
    args: list[str],
    *,
    duration_s: float,
    timeout_s: float,
    progress: Progress | None = None,
    span: tuple[float, float] = (0.0, 1.0),
    message: str = "",
) -> None:
    """Run ffmpeg with -progress on stdout, mapping out_time to ``progress`` within ``span``."""
    cmd = [get_settings().ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    cmd += args[:-1] + ["-progress", "pipe:1", "-nostats", args[-1]]
    with tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        except FileNotFoundError as exc:
            raise IngestError(
                "ffmpeg isn't installed, so videos can't be converted. Run `hoops doctor`."
            ) from exc
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout_s, kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and progress and value.isdigit() and duration_s > 0:
                    frac = min(1.0, int(value) / 1e6 / duration_s)
                    progress(span[0] + (span[1] - span[0]) * frac, message)
            code = proc.wait()
        except BaseException:
            proc.kill()  # cancelled from the progress callback, or interrupted
            proc.wait()
            raise
        finally:
            timer.cancel()
        if code != 0:
            if timed_out.is_set():
                raise IngestError("Converting the video took too long and was stopped.")
            err.seek(0)
            tail = err.read().decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
            raise IngestError(f"We couldn't convert this video (ffmpeg: {tail[0][:300]}).")


# ---------------------------------------------------------------- outputs


def _asset(path: Path, store: Store) -> VideoAsset:
    info = ffprobe_json(path)
    vs = video_stream(info) or {}
    rate = vs.get("r_frame_rate") or "30/1"
    fps_float = float(_rate(rate) or 30)
    duration = _duration(info, vs)
    n_frames = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else round(duration * fps_float)
    return VideoAsset(
        path=store.rel(path),
        width=int(vs.get("width", 0)),
        height=int(vs.get("height", 0)),
        fps=rate,
        fps_float=round(fps_float, 6),
        n_frames=n_frames,
        duration_s=round(duration, 6),
        codec=vs.get("codec_name", "unknown"),
        pix_fmt=vs.get("pix_fmt", "unknown"),
        has_audio=any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
    )


def _proxy_fps(master_fps: str) -> str:
    f = Fraction(master_fps)
    return master_fps if f <= Fraction(30) else str(f / 2)


def _poster(master: Path, out: Path, t_s: float) -> None:
    cmd = [get_settings().ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    cmd += ["-ss", f"{t_s:.3f}", "-i", str(master), "-frames:v", "1", "-update", "1", "-q:v", "3"]
    cmd += ["-vf", "scale='min(iw,1280)':'min(ih,1280)':force_original_aspect_ratio=decrease", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _cached(store: Store, clip_id: str) -> ClipMeta | None:
    try:
        clip = store.load_clip(clip_id)
    except Exception:  # noqa: BLE001 - a stale/corrupt cache entry is simply rebuilt
        return None
    if clip and store.abs(clip.master.path).exists() and store.abs(clip.proxy.path).exists():
        return clip
    return None


def ingest(
    src: Path,
    *,
    original_filename: str | None = None,
    store: Store | None = None,
    progress: Progress | None = None,
) -> ClipMeta:
    """Normalize an uploaded video (or return the cached result for identical bytes)."""
    src = Path(src)
    store = store or Store()
    if not src.is_file():
        raise IngestError("The uploaded file is missing.")

    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    tick(0.0, "Checking the video")
    clip_id = sha256_file(src)[:16]
    if (hit := _cached(store, clip_id)) is not None:
        tick(1.0, "Already prepared")
        return hit

    info = probe(src)
    try:
        caps = ffmpeg_capabilities()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise IngestError(
            "ffmpeg isn't installed, so videos can't be converted. Run `hoops doctor`."
        ) from exc
    interlaced = detect_interlaced(src) if needs_idet(info) else None
    plan = plan_normalization(info, caps.filters, interlaced=interlaced)
    plan.report.ffmpeg_version = caps.version

    d = store.clip_dir(clip_id)
    ext = re.sub(r"[^a-z0-9]", "", src.suffix.lower())[:5] or "bin"
    source = d / f"source.{ext}"
    if src.resolve() != source.resolve():
        shutil.copyfile(src, source)
    (d / "probe.json").write_text(json.dumps(info, indent=2))

    master, proxy, poster = d / "master.mp4", d / "proxy_720p.mp4", d / "poster.jpg"
    tmp_master, tmp_proxy = d / ".master.tmp.mp4", d / ".proxy.tmp.mp4"
    gop = max(1, round(2 * float(Fraction(plan.fps))))
    args = [*plan.input_args, "-protocol_whitelist", "file", "-i", str(source)]
    args += ["-map", "0:v:0", "-map", "0:a:0?", "-dn", "-sn", "-map_metadata", "-1", "-map_chapters", "-1"]
    args += ["-vf", ",".join(plan.filters), "-fps_mode", "cfr"]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
    ]
    args += ["-g", str(gop), "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    args += ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000", "-max_muxing_queue_size", "1024"]
    args += ["-movflags", "+faststart", str(tmp_master)]
    timeout = 300 + 30 * plan.duration_s
    try:
        _run_ffmpeg(
            args,
            duration_s=plan.duration_s,
            timeout_s=timeout,
            progress=progress,
            span=(0.02, _P_MASTER),
            message="Converting video",
        )
        os.replace(tmp_master, master)

        master_fps = (video_stream(ffprobe_json(master)) or {}).get("r_frame_rate") or plan.fps
        pfps = _proxy_fps(master_fps)
        args = ["-protocol_whitelist", "file", "-i", str(master), "-map", "0:v:0", "-an", "-sn", "-dn"]
        args += [
            "-map_metadata",
            "-1",
            "-vf",
            f"{_fit(1280, 720)},fps={pfps},format=yuv420p",
            "-fps_mode",
            "cfr",
        ]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
        args += ["-g", str(max(1, round(float(Fraction(pfps))))), "-movflags", "+faststart", str(tmp_proxy)]
        _run_ffmpeg(
            args,
            duration_s=plan.duration_s,
            timeout_s=timeout,
            progress=progress,
            span=(_P_MASTER, _P_PROXY),
            message="Making the analysis copy",
        )
        os.replace(tmp_proxy, proxy)
    finally:
        tmp_master.unlink(missing_ok=True)
        tmp_proxy.unlink(missing_ok=True)

    tick(_P_PROXY, "Finishing up")
    master_asset, proxy_asset = _asset(master, store), _asset(proxy, store)
    for asset in (master_asset, proxy_asset):
        if asset.codec != "h264" or asset.pix_fmt != "yuv420p":
            raise IngestError("The converted video failed its final check; please try another file.")
    try:
        _poster(master, poster, master_asset.duration_s * 0.1)
    except (subprocess.SubprocessError, OSError) as exc:
        plan.report.warnings.append(f"No poster image ({type(exc).__name__}).")

    cuts: list[float] = []
    try:
        from hoops.sampling import scene_cuts

        cuts = scene_cuts(proxy)
    except ImportError:
        plan.report.warnings.append("Scene-cut detection skipped (install the `cv` extra).")
    except Exception as exc:  # noqa: BLE001 - cuts are an optimization; never fail the ingest for them
        plan.report.warnings.append(f"Scene-cut detection failed ({type(exc).__name__}).")

    plan.report.summary = summarize(plan.report, master_asset)
    clip = ClipMeta(
        clip_id=clip_id,
        original_filename=original_filename or src.name,
        source_size_bytes=src.stat().st_size,
        master=master_asset,
        proxy=proxy_asset,
        normalization=plan.report,
        scene_cuts_s=cuts,
    )
    store.save_clip(clip)
    tick(1.0, plan.report.summary)
    return clip
