from __future__ import annotations

import subprocess
from fractions import Fraction

import pytest

from hoops import config
from hoops import ingest as ing
from hoops.ingest import IngestError, ingest, plan_normalization, snap_fps
from hoops.store import Store
from hoops.video_io import ffprobe_json, is_browser_playable
from tests.helpers import make_video, requires_ffmpeg

pytestmark = requires_ffmpeg


def _video_stream(path):
    return next(s for s in ffprobe_json(path)["streams"] if s["codec_type"] == "video")


def _assert_normalized(store: Store, clip) -> None:
    for asset in (clip.master, clip.proxy):
        p = store.abs(asset.path)
        assert p.exists() and is_browser_playable(p)
        vs = _video_stream(p)
        assert vs["pix_fmt"] == "yuv420p" and Fraction(vs["r_frame_rate"]) == Fraction(vs["avg_frame_rate"])
        assert not any(
            "matrix" in str(d.get("side_data_type", "")).lower() for d in vs.get("side_data_list", [])
        )
    assert store.abs(clip.master.path).parent.joinpath("poster.jpg").exists()
    assert store.load_clip(clip.clip_id) == clip


def test_landscape_roundtrip(data_dir, tmp_path):
    src = make_video(tmp_path / "game.mov", duration=1.5)
    seen: list[float] = []
    clip = ingest(src, original_filename="Game 1.mov", progress=lambda f, m: seen.append(f))
    store = Store()
    _assert_normalized(store, clip)
    assert (clip.master.width, clip.master.height) == (640, 360)
    assert clip.master.fps == "30/1" and clip.master.has_audio and not clip.proxy.has_audio
    assert abs(clip.master.duration_s - 1.5) < 0.1 and clip.master.n_frames == 45
    assert clip.original_filename == "Game 1.mov" and len(clip.clip_id) == 16
    assert clip.master.path == f"clips/{clip.clip_id}/master.mp4"
    assert (data_dir / "clips" / clip.clip_id / "source.mov").exists()
    assert (data_dir / "clips" / clip.clip_id / "probe.json").exists()
    assert clip.normalization.summary == "30 fps constant frame rate, 360p"
    assert clip.normalization.ffmpeg_version and clip.scene_cuts_s == []
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_rotated_phone_clip_comes_out_upright(data_dir, tmp_path):
    flat = make_video(tmp_path / "flat.mp4", duration=1.0, audio=False)
    rotated = tmp_path / "portrait.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation:v:0", "90", "-i", str(flat)]
        + ["-c", "copy", str(rotated)],
        check=True,
    )
    clip = ingest(rotated)
    _assert_normalized(Store(), clip)
    assert (clip.master.width, clip.master.height) == (360, 640)
    assert (clip.proxy.width, clip.proxy.height) == (360, 640)
    assert clip.normalization.rotation_deg in (90, 270)
    assert clip.normalization.summary.startswith("Rotated 90°") and "portrait" in clip.normalization.summary


def test_vfr_becomes_cfr(data_dir, tmp_path):
    src = make_video(
        tmp_path / "vfr.mp4",
        duration=3.0,
        audio=False,
        extra_out=["-vf", r"select='lt(n\,30)+not(mod(n\,3))'", "-fps_mode", "vfr"],
    )
    vs = _video_stream(src)
    assert Fraction(vs["r_frame_rate"]) != Fraction(vs["avg_frame_rate"])
    clip = ingest(src)
    _assert_normalized(Store(), clip)
    assert clip.normalization.vfr_source and clip.normalization.fps_snapped_from
    assert clip.master.fps in ing.STANDARD_FPS or clip.master.fps in {"24/1", "25/1", "30/1", "50/1", "60/1"}
    assert "(was variable)" in clip.normalization.summary


def test_hdr_10bit_is_tonemapped_to_sdr(data_dir, tmp_path):
    src = make_video(
        tmp_path / "hlg.mp4",
        duration=1.0,
        audio=False,
        pix_fmt="yuv420p10le",
        extra_out=["-vf", "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"],
    )
    assert _video_stream(src)["color_transfer"] == "arib-std-b67"
    clip = ingest(src)
    _assert_normalized(Store(), clip)
    vs = _video_stream(Store().abs(clip.master.path))
    assert vs["color_transfer"] == "bt709" and vs["color_primaries"] == "bt709"
    assert clip.normalization.hdr == "hlg"
    caps = ing.ffmpeg_capabilities()
    if {"zscale", "tonemap"} <= caps.filters or "scale_vt" in caps.filters:
        assert clip.normalization.tonemapped and "HDR→SDR" in clip.normalization.summary
    else:
        assert clip.normalization.warnings


@pytest.mark.skipif("scale_vt" not in ing.ffmpeg_capabilities().filters, reason="needs VideoToolbox")
def test_hdr_fallback_without_zscale_uses_videotoolbox(data_dir, tmp_path, monkeypatch):
    real = ing.ffmpeg_capabilities()
    fake = ing.FFmpegCaps(real.version, real.filters - {"zscale"}, real.encoders)
    monkeypatch.setattr(ing, "ffmpeg_capabilities", lambda *a: fake)
    src = make_video(
        tmp_path / "pq.mp4",
        duration=1.0,
        audio=False,
        pix_fmt="yuv420p10le",
        extra_out=["-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc"],
    )
    clip = ingest(src)
    assert clip.normalization.hdr == "pq" and clip.normalization.tonemapped
    assert any("VideoToolbox" in w for w in clip.normalization.warnings)
    assert _video_stream(Store().abs(clip.master.path))["pix_fmt"] == "yuv420p"


def test_interlaced_is_deinterlaced(data_dir, tmp_path):
    src = make_video(
        tmp_path / "broadcast.mp4",
        duration=1.0,
        audio=False,
        extra_out=["-vf", "setfield=tff", "-flags", "+ildct+ilme"],
    )
    assert _video_stream(src)["field_order"] in {"tt", "tb"}
    clip = ingest(src)
    _assert_normalized(Store(), clip)
    assert clip.normalization.deinterlaced and clip.normalization.summary.startswith("Deinterlaced")


def test_cache_hit_returns_stored_clip(data_dir, tmp_path):
    src = make_video(tmp_path / "a.mp4", duration=1.0, audio=False)
    first = ingest(src)
    messages: list[str] = []
    again = ingest(src, progress=lambda f, m: messages.append(m))
    assert again == first and messages[-1] == "Already prepared"


def test_non_video_is_rejected(data_dir, tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("definitely not a video")
    with pytest.raises(IngestError, match="doesn't look like a video"):
        ingest(bogus)
    assert not (data_dir / "clips").exists() or not any((data_dir / "clips").iterdir())


def test_too_long_is_rejected(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HOOPS_MAX_CLIP_SECONDS", "1")
    config.get_settings.cache_clear()
    src = make_video(tmp_path / "long.mp4", duration=2.0, audio=False)
    with pytest.raises(IngestError, match="limit"):
        ingest(src)


# ---------------------------------------------------------------- decision table (no ffmpeg runs)


def _info(**video):
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "field_order": "progressive",
        **video,
    }
    return {"format": {"format_name": "mov,mp4", "duration": "10.0"}, "streams": [stream]}


ALL_FILTERS = {"zscale", "tonemap", "bwdif", "scale_vt"}


def test_plan_rejects_dolby_vision_profile_5():
    info = _info(side_data_list=[{"side_data_type": "DOVI configuration record", "dv_profile": 5}])
    with pytest.raises(IngestError, match="Dolby Vision profile 5"):
        plan_normalization(info, ALL_FILTERS)


def test_plan_dolby_vision_81_tonemaps_base_layer():
    info = _info(
        color_transfer="arib-std-b67",
        side_data_list=[{"side_data_type": "DOVI configuration record", "dv_profile": 8}],
    )
    plan = plan_normalization(info, ALL_FILTERS)
    assert plan.report.hdr == "dolby_vision" and plan.report.tonemapped
    assert any(f.startswith("zscale=tin=arib-std-b67") for f in plan.filters)


def test_plan_rotation_sar_and_legacy_tags():
    plan = plan_normalization(
        _info(tags={"rotate": "90"}, sample_aspect_ratio="4:3", width=720, height=480), []
    )
    assert plan.report.rotation_deg == 270 and plan.report.sar_fixed
    assert plan.report.source_resolution == (480, 960)
    assert "setsar=1" in plan.filters


def test_plan_without_tonemap_filters_warns():
    plan = plan_normalization(_info(color_transfer="smpte2084"), {"bwdif"})
    assert plan.report.hdr == "pq" and not plan.report.tonemapped and plan.report.warnings


def test_plan_idet_verdict_forces_deinterlace():
    plan = plan_normalization(_info(field_order="unknown"), ALL_FILTERS, interlaced=True)
    assert plan.report.deinterlaced and plan.filters[0].endswith("deint=all")
    assert ing.needs_idet(_info(field_order="unknown")) and not ing.needs_idet(_info())


def test_plan_sd_matrix_converted_and_photo_rejected():
    plan = plan_normalization(_info(color_space="smpte170m"), ALL_FILTERS)
    assert any("out_color_matrix=bt709" in f for f in plan.filters)
    photo = _info()
    photo["format"]["format_name"] = "png_pipe"
    with pytest.raises(IngestError, match="photo"):
        plan_normalization(photo, ALL_FILTERS)


@pytest.mark.parametrize(
    ("fps", "want"),
    [(29.97, "30000/1001"), (29.83, "30000/1001"), (25.0, "25"), (59.94, "60000/1001"), (119.88, "60000/1001"),
     (240.0, "60"), (23.976, "24000/1001"), (48.0, "50")],
)  # fmt: skip
def test_snap_fps(fps, want):
    assert snap_fps(fps) == want
