"""Detector benchmark: ms/frame and FPS per detector and input size on this machine (warm-up excluded).

Uses frames from ``clip`` when given (decode time excluded), else synthetic 720p frames.
"""

from __future__ import annotations

import importlib
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hoops import registry


def _frames(clip: Path | None, n: int) -> list[np.ndarray]:
    if clip is not None:
        from hoops.frames import iter_frames

        out: list[np.ndarray] = []
        for _, rgb in iter_frames(clip):
            out.append(rgb)
            if len(out) >= n:
                break
        if out:
            return out
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):  # a wooden floor with a few moving player-sized blocks
        f = np.full((720, 1280, 3), (150, 120, 90), dtype=np.uint8)
        for k in range(6):
            x = (100 + 180 * k + 7 * i) % 1200
            f[300:520, x : x + 60] = rng.integers(0, 255, 3, dtype=np.uint8)
        frames.append(f)
    return frames


def bench(
    detector_keys: Sequence[str] = ("rfdetr:coco",),
    sizes: Sequence[int] = (640, 1280),
    n_frames: int = 30,
    clip: Path | None = None,
    warmup: int = 3,
) -> list[dict[str, Any]]:
    """One row per (detector, size): ms_per_frame, fps, device, version; ``error`` when it couldn't run."""
    importlib.import_module("hoops.perception.detectors")  # registers detectors
    frames = _frames(clip, n_frames)
    machine = f"{platform.system()} {platform.machine()}"
    rows: list[dict[str, Any]] = []
    for key in detector_keys:
        try:
            det = registry.detectors.create(key)
        except Exception as exc:  # noqa: BLE001 - report, keep benchmarking the rest
            rows += [{"detector": key, "size": s, "error": f"{type(exc).__name__}: {exc}"} for s in sizes]
            continue
        for size in sizes:
            row: dict[str, Any] = {"detector": key, "size": size, "n_frames": len(frames), "machine": machine}
            try:
                if hasattr(det, "resolution"):
                    det.resolution = size
                else:
                    det = registry.detectors.create(key, resolution=size)
                for f in frames[:warmup]:
                    det.detect(f)
                t0 = time.perf_counter()
                for f in frames:
                    det.detect(f)
                dt = time.perf_counter() - t0
                info = det.info()
                row.update(
                    ms_per_frame=round(1000 * dt / len(frames), 1),
                    fps=round(len(frames) / dt, 1),
                    device=info.get("device"),
                    version=info.get("version"),
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [f"{'detector':<22} {'size':>5} {'device':>6} {'ms/frame':>9} {'FPS':>6}"]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['detector']:<22} {r['size']:>5}  error: {r['error']}")
        else:
            lines.append(
                f"{r['detector']:<22} {r['size']:>5} {r.get('device') or '?':>6} "
                f"{r['ms_per_frame']:>9.1f} {r['fps']:>6.1f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_table(bench()))
